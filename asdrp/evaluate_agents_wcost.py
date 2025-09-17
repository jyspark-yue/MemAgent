#############################################################################
# File: evaluate_agents_wcost.py
#
# Description:
#   Builds on "evaluate_reductive_agent.py" by adding tracking on token usage, real-world cost, and latency.
#
# Authors:
#   @author     Varenya Garg
#               Created evaluate_reductive_agent.py; simulates agent memory usage by replaying chat history (haystack_sessions)
#               and measuring how well the agent answers long-term memory questions after replaying the chat history.
#   @author     Oliver Hsu
#               Implemented cost tracking in evaluate_reductive_agent.py.
#
# Contributors:
#   @contributor    Eric Vincent Fernandes
#                   Optimized evaluate_agents_wcost.py for faster runtime:
#                   (Brought down eval runtime from ~18 mins for 1 question to ~X mins for 500 questions [Dataset: longmemeval_m.json, ReductiveAgent])
#
# Date:
#   Created:    August 3, 2025  (Varenya Garg)
#   Modified:   August 22, 2025 (Oliver Hsu)
#   Modified:   September 3, 2025 (Eric Vincent Fernandes)
#############################################################################

import asyncio
import json
import os
import time
import traceback

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI

# IMPORTANT: Import the custom agent to evaluate
# from asdrp.agent.summary_agent import SummaryAgent
from asdrp.agent.reductive_agent import ReductiveAgent

# from asdrp.agent.episodic_agent import EpisodicAgent
# from asdrp.agent.hierarchical_vector_agent import HVMAgent

# Import the OpenAI RateLimitError to detect rate-limit exceptions
try:
    from openai.error import RateLimitError
except ImportError:
    # Fallback definition in case openai package is not available
    class RateLimitError(Exception):
        """Fallback RateLimitError used when openai.error isn't importable at analysis time."""
        pass

# Constants to compute approximate cost for API usage (o4-mini)
INPUT_COST_PER_1K = 0.0011  # Cost per 1000 input tokens ($)    [$1.10 * (1000/1000000)]
OUTPUT_COST_PER_1K = 0.0044  # Cost per 1000 output tokens ($)   [$4.40 * (1000/1000000)]

# Retry configuration for rate-limit handling
RETRY_ATTEMPTS = 5      # Maximum number of retry attempts on RateLimitError
RETRY_BASE_DELAY = 10  # Base seconds for exponential backoff between retries


async def write_results_to_file(output_file, results, append=False):
    """
    Write evaluation results to a JSONL file (one JSON object per line).
    Creates the output directory if it does not exist.

    Args:
        output_file (str): Path to save results
        results (list[dict]): List of result dictionaries
        append (bool): Append mode if True, overwrite if False
    """

    mode = 'a' if append else 'w'
    print(f"Saving results to {output_file} (append={append})...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode) as f:
        for result in results:
            # ensure we always write a json object per line
            f.write(json.dumps(result) + "\n")
    print(f"Results saved in {output_file}")


async def load_chat_history(agent_object, haystack_sessions):
    """
    Replay chat history into the agent's memory block.
    Each session contains turns of user and assistant messages.

    Args:
        agent_object: The agent instance whose memory is populated
        haystack_sessions (list[list[dict]]): List of chat sessions
    """

    print(f"Processing {len(haystack_sessions)} haystack sessions...")

    session_count = 0
    for session in haystack_sessions:
        session_count += 1
        if session_count % 5 == 0:  # Print progress every 5 sessions
            print(f"Processed {session_count}/{len(haystack_sessions)} sessions...")

        user_msg = None
        agent_text = None

        for turn in session:

            # Clean content to avoid tokenizer special-token errors
            content = turn["content"].replace("<|endoftext|>", "")

            # Separate user and assistant messages
            if turn["role"] == "user":
                user_msg = ChatMessage(role="user", content=content)
            elif turn["role"] == "assistant":
                agent_text = ChatMessage(role="assistant", content=content)

                # Only push user-assistant pairs
                if user_msg is not None and agent_text is not None:
                    try:
                        await agent_object.memory_block._aput(
                            [user_msg, agent_text]) # IMPORTANT: memory_block variable is constant across agents
                    except Exception as e:
                        print(f"Error processing turn: {e}")
                        traceback.print_exc()
                        continue

                    # Reset temporary variables for next pair
                    user_msg = None
                    agent_text = None


def reset_memory(agent_object):
    """
    Reset the agent's memory and reinitialize the agent instance.
    Useful before processing a new question to avoid contamination
    from previous sessions.
    """

    agent_object.memory = agent_object._create_memory()
    agent_object.agent = agent_object._create_agent(agent_object.memory, [])


class LongMemEvalRunner:
    """
    Class to run evaluation on a dataset of long-term memory questions.
    Handles token tracking, cost calculation, rate-limit retry, and parallelism.
    """

    def __init__(self, agent, callback_manager, handler):
        self.agent = agent
        self.callback_manager = callback_manager
        self.handler = handler
        self.semaphore = asyncio.Semaphore(10)   # Sets limit to 3 questions processed at a time

    async def process_question(self, question, question_num):
        """
        Evaluate a single question: load history, query the agent, compute token usage & cost.

        Args:
            question (dict): Contains question text, ID, and haystack_sessions
            question_num (int): Index in the current batch

        Returns:
            dict: Results including tokens, cost, response, time, and error (if any)
        """

        print(f"Processing question {question_num}: {question.get('question_id', 'unknown')}")

        # Check the structure of the item
        print(f"Item keys: {list(question.keys())}")
        print(f"Number of haystack sessions: {len(question.get('haystack_sessions', []))}")

        # Reset agent memory for this question
        agent_object = create_agent(self.agent, self.callback_manager)
        print("Memory reset, processing chat history...")

        lch_time = lch_input_tokens = lch_output_tokens = lch_cost = 0
        query_time = query_input_tokens = query_output_tokens = query_cost = 0

        try:
            # Replay all haystack sessions into memory
            await load_chat_history(agent_object, question['haystack_sessions'])

            lch_input_tokens = self.agent.memory_block.input_tokens
            lch_output_tokens = self.agent.memory_block.output_tokens
            lch_time = self.agent.memory_block.load_chat_history_time

            lch_cost = (lch_input_tokens / 1000.0) * INPUT_COST_PER_1K + (lch_output_tokens / 1000.0) * OUTPUT_COST_PER_1K

            # REMOVE ALL PROMPT STUFF ANMD MOVE AROUND THE LLM CALL, OVER HER CREATE VARS AND USE GLOBAL FIELDS TO ACCESS agent_object.memory_block.xyz

            # Measure tokens used by memory block/agent
            # initial_query_input_tokens = self.handler.prompt_llm_token_count
            # initial_query_output_tokens = self.handler.completion_llm_token_count
            # initial_query_time = time.time()

            # Retry mechanism for rate-limit errors
            last_exc = None
            answer_text = None
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    response = await agent_object.achat(question['question'])
                    answer_text = response.response_str
                    print(f"Got response: {answer_text[:100]}...")  # Show first 100 chars
                    last_exc = None
                    break
                except RateLimitError as e:
                    last_exc = e
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"RateLimitError attempt {attempt}/{RETRY_ATTEMPTS}: {e}. Backing off {delay}s...")
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-rate-limit errors: surface them immediately
                    raise

            if last_exc is not None and answer_text is None:
                # Still failing rate-limit after retries
                raise last_exc

            # Compute query tokens and cost for this question
            # query_time = time.time() - initial_query_time
            # query_input_tokens = self.handler.prompt_llm_token_count - initial_query_input_tokens
            # query_output_tokens = self.handler.completion_llm_token_count - initial_query_output_tokens
            # query_cost = (query_input_tokens / 1000.0) * INPUT_COST_PER_1K + (query_output_tokens / 1000.0) * OUTPUT_COST_PER_1K

            query_input_tokens = self.agent.query_input_tokens
            query_output_tokens = self.agent.query_output_tokens
            query_time = self.agent.query_time
            query_cost = (query_input_tokens / 1000.0) * INPUT_COST_PER_1K + (query_output_tokens / 1000.0) * OUTPUT_COST_PER_1K

        except Exception as e:
            print(f"Error processing question {question_num + 1}: {e}")
            traceback.print_exc()
            answer_text = f"Error: {str(e)}"

        overall_input_tokens = lch_input_tokens + query_input_tokens
        overall_output_tokens = lch_output_tokens + query_output_tokens
        overall_cost = lch_cost + query_cost
        overall_time = lch_time + query_time

        result = {
            "question_id": question["question_id"],
            "hypothesis": answer_text,
            "memory_input_tokens": lch_input_tokens,
            "memory_output_tokens": lch_output_tokens,
            "memory_cost": lch_cost,
            "memory_time": lch_time,
            "query_input_tokens": query_input_tokens,
            "query_output_tokens": query_output_tokens,
            "query_cost": query_cost,
            "query_time": query_time,
            "question_input_tokens": overall_input_tokens,
            "question_output_tokens": overall_output_tokens,
            "question_cost": overall_cost,
            "question_time": overall_time
        }

        if isinstance(answer_text, str) and answer_text.startswith("Error:"):
            result["error"] = answer_text
            print(f"ERROR={answer_text}")
        else:
            print(f"[Q{question_num + 1}] overall_input_tokens={overall_input_tokens}  overall_output_tokens={overall_output_tokens}  cost=${overall_cost:.6f}  time={overall_time:.2f}s")

        return result

    async def process_question_worker(self, item, i):
        """
        Worker wrapper: manage semaphore acquisition, run question processing, handle exceptions.

        Args:
            item (dict): Question item
            i (int): Index in dataset

        Returns:
            dict: Result dictionary including error info if exceptions occur
        """

        acquired = False
        try:
            # Acquire semaphore with timeout
            await asyncio.wait_for(self.semaphore.acquire(), timeout=3600)
            acquired = True
        except asyncio.TimeoutError:
            # Return an error if semaphore acquisition times out
            return {
                "question_id": item.get("question_id", f"idx_{i}"),
                "hypothesis": f"Error: timed out waiting for semaphore",
                "memory_input_tokens": 0,
                "memory_output_tokens": 0,
                "memory_cost": 0.0,
                "memory_time": 0.0,
                "query_input_tokens": 0,
                "query_output_tokens": 0,
                "query_cost": 0.0,
                "query_time": 0.0,
                "question_input_tokens": 0,
                "question_output_tokens": 0,
                "question_cost": 0.0,
                "question_time": 0.0,
                "error": "semaphore_timeout"
            }

        try:
            # Process the question normally
            result = await self.process_question(item, i)
            return result
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "question_id": item.get("question_id", f"idx_{i}"),
                "hypothesis": f"Error: {str(e)}",
                "memory_input_tokens": 0,
                "memory_output_tokens": 0,
                "memory_cost": 0.0,
                "memory_time": 0.0,
                "query_input_tokens": 0,
                "query_output_tokens": 0,
                "query_cost": 0.0,
                "query_time": 0.0,
                "question_input_tokens": 0,
                "question_output_tokens": 0,
                "question_cost": 0.0,
                "question_time": 0.0,
                "error": "exception",
                "traceback": tb
            }
        finally:
            # Releases the semaphore only if acquired
            if acquired:
                try:
                    self.semaphore.release()
                except Exception:
                    pass

    async def evaluate_on_dataset(self, data_file, output_file, num_questions, start_index=0):
        """
        Run evaluation on a dataset of questions in parallel, respecting semaphore limits.
        Writes results to file and prints overall cost & time.

        Args:
            data_file (str): JSON dataset path
            output_file (str): File to save results
            num_questions (int): Number of questions to process
            start_index (int): Index in dataset to start processing
        """

        print(f"Loading dataset from {data_file}...")
        with open(data_file, 'r') as f:
            dataset = json.load(f)

        print(f"Dataset loaded with {len(dataset)} total questions")

        # Slice dataset for partial evaluation runs
        if num_questions is not None:
            dataset = dataset[start_index:start_index + num_questions]
            print(f"Processing questions [{start_index}:{start_index + num_questions}] "
                  f"(total this run: {len(dataset)})")
        else:
            dataset = dataset[start_index:]
            print(f"Processing questions [{start_index}:] (total this run: {len(dataset)})")

        print(f"Running {len(dataset)} questions in parallel...")

        # Record starting token counts and time for overall metrics for the entire dataset
        overall_start_time = time.time()
        # overall_prompt0 = self.handler.prompt_llm_token_count
        # overall_comp0 = self.handler.completion_llm_token_count

        # Kick off all workers in parallel, semaphore limits to 3 at a time
        tasks = [asyncio.create_task(self.process_question_worker(item, i)) for i, item in enumerate(dataset)]
        results = await asyncio.gather(*tasks)

        # Determine append mode based on start index
        append_mode = start_index > 0
        await write_results_to_file(output_file, results, append=append_mode)

        # # Calculate overall token usage and cost
        # overall_prompt = self.handler.prompt_llm_token_count - overall_prompt0
        # overall_comp = self.handler.completion_llm_token_count - overall_comp0
        # overall_time = time.time() - overall_t0
        # overall_cost = (overall_prompt / 1000.0) * INPUT_COST_PER_1K + (overall_comp / 1000.0) * OUTPUT_COST_PER_1K

        total_memory_input_tokens = sum(r.get("memory_input_tokens", 0) for r in results)
        total_memory_output_tokens = sum(r.get("memory_output_tokens", 0) for r in results)
        total_memory_cost = sum(r.get("memory_cost", 0.00) for r in results)

        total_query_input_tokens = sum(r.get("query_input_tokens", 0) for r in results)
        total_query_output_tokens = sum(r.get("query_output_tokens", 0) for r in results)
        total_query_cost = sum(r.get("query_cost", 0.00) for r in results)

        total_combined_input_tokens = sum(r.get("question_input_tokens", 0) for r in results)
        total_combined_output_tokens = sum(r.get("question_output_tokens", 0) for r in results)
        total_combined_cost = sum(r.get("question_cost", 0) for r in results)

        mean_combined_input_tokens = total_combined_input_tokens / len(dataset)
        mean_combined_output_tokens = total_combined_output_tokens / len(dataset)
        mean_combined_cost = total_combined_cost / len(dataset)

        if (total_combined_input_tokens != (total_memory_input_tokens + total_query_input_tokens)):
            print("CALCULATION DIFFERENCE DETECTED!!!!!!!!!!")

        # Combined totals (from per-question sums)
        # combined_prompt = total_memory_input_tokens + total_query_input_tokens
        # combined_comp = total_memory_output_tokens + total_query_output_tokens
        # combined_cost = total_memory_cost + total_query_cost

        overall_time = time.time() - overall_start_time

        # print("\n=== Run summary ===")
        # print(f"Total prompt tokens:     {overall_prompt}")
        # print(f"Total completion tokens: {overall_comp}")
        # print(f"Total tokens:            {overall_prompt + overall_comp}")
        # print(f"Estimated total cost:    ${overall_cost:.6f}")
        # print(f"Total wall time:         {overall_time:.2f}s")

        print("\n=== Run summary ===")
        print(f"Memory loading prompt tokens:           {total_memory_input_tokens}")
        print(f"Memory loading completion tokens:       {total_memory_output_tokens}")
        print(f"Memory loading cost:                    ${total_memory_cost:.6f}")
        print(f"Query prompt tokens:                    {total_query_input_tokens}")
        print(f"Query completion tokens:                {total_query_output_tokens}")
        print(f"Query cost:                             ${total_query_cost:.6f}")
        print(f"Total prompt tokens:                    {total_combined_input_tokens}")
        print(f"Total completion tokens:                {total_combined_output_tokens}")
        print(f"Total tokens:                           {total_combined_input_tokens + total_combined_output_tokens}")
        print(f"Total cost:                             ${total_combined_cost:.6f}")
        print(f"Average prompt tokens per question:     {mean_combined_input_tokens}")
        print(f"Average completion tokens per question: {mean_combined_output_tokens}")
        print(f"Average total cost per question:        ${mean_combined_cost:.6f}")
        print(f"Total evaluation time:                  {overall_time:.4f}s")


def create_agent(agent_class, callback_manager=None):
    """
    Create an instance of the agent with LLM and token tracker attached.

    Args:
        agent_class: The class of the agent to instantiate
        callback_manager: Callback manager for token tracking

    Returns:
        agent_instance: Initialized agent
    """

    llm = OpenAI(model="o4-mini", callback_manager=CallbackManager(handlers=[TokenCountingHandler()]))
                 # callback_manager=callback_manager)
    agent_instance = agent_class(llm=llm)
    return agent_instance


def main():
    """
    Entry point for evaluation script. Configures handler, callback manager,
    agent runner, dataset, and output paths. Runs evaluation asynchronously.
    """

    start_index = 0
    num_questions = 500

    handler = TokenCountingHandler()
    cb_manager = CallbackManager(handlers=[handler])

    runner = LongMemEvalRunner(ReductiveAgent, cb_manager, handler)   # IMPORTANT: CHANGE BASED ON AGENT

    # Paths for dataset and results
    data_file = "eval/data/custom_history/longmemeval_m.json"
    output_file = "results/reductive_agent_responses.json"            # IMPORTANT: CHANGE BASED ON AGENT

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    # Run evaluation asynchronously
    asyncio.run(
        runner.evaluate_on_dataset(data_file, output_file, start_index=start_index, num_questions=num_questions))


if __name__ == "__main__":
    main()
