#############################################################################
# File: evaluate_agents_wcost.py
#
# Description:
#   Builds on "evaluate_reductive_agent.py" by adding tracking on token usage, real-world cost, and latency.
#
# Authors:
#   @author     Varenya Garg
#               - Created evaluate_reductive_agent.py
#               - Set up simulation of agent memory-usage by replaying chat history (haystack_sessions)
#               - Saved results (question input, hypothesis) to a JSON file
#   @author     Oliver Hsu
#               - Implemented cost tracking in evaluate_reductive_agent.py.
#   @author     Eric Vincent Fernandes
#               - Changed implementation to be run in parallel instead of sequential
#               - Added a semaphore and batched question_workers to prevent RateLimitErrors
#               - Added more metrics for capturing token cost for more detailed insights
#               - Moved token collection from evaluate_agents_wcost.py to the agent/memory pairs for safer retrieval.
#               - Adjusted input/output token costs to be accurate for our model (gpt-5-nano)
#               - Grouped inputs for PropositionalExtractionMemory by session to reduce LLM-calls and to store information in context of the overall conversation
#               - Brought down eval runtime from ~18 mins for 1 question to ~8 mins for 500 questions (Dataset: longmemeval_m.json, CondensedMemoryBlock)
#               - Modified dataset path-locator to be more general
#               - Saved results for each question to JSON file once acquired, instead of at the end, for safe record keeping
#               - Added functionality to save token metrics to a JSON file after runtime
#
# Date:
#   Created:    August 3, 2025  (Varenya Garg)
#   Modified:   August 22, 2025 (Oliver Hsu)
#   Modified:   September 21, 2025 (Eric Vincent Fernandes)
#############################################################################

import asyncio
import json
import os
import time
import traceback
from typing import List

from llama_index.core.base.llms.types import ChatMessage

from asdrp.agent.summary_agent import SummaryAgent
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

# Constants to compute approximate cost for API usage (gpt-5-nano)
INPUT_COST_PER_1K = 0.00005  # Cost per 1000 input tokens ($)    [$0.05 * (1000/1000000)]
OUTPUT_COST_PER_1K = 0.00040  # Cost per 1000 output tokens ($)   [$0.40 * (1000/1000000)]

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
            f.write(json.dumps(result) + "\n")  # ensures there is only one json object per line
    print(f"Results saved in {output_file}")


def load_completed_ids(output_file):
    """
    Load completed ids from a JSONL file (one JSON object per line).
    Creates the output directory if it does not exist.

    Args:
        output_file (str): Path to save results
    """
    if not os.path.exists(output_file):
        return set()
    completed = set()
    with open(output_file, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                completed.add(obj["question_id"])
            except Exception:
                continue
    return completed


async def load_chat_history(agent_object, haystack_sessions):
    """
    Replay chat history into the agent's memory block.
    Each session contains turns of user and assistant messages.

    Args:
        agent_object: The agent instance whose memory is populated
        haystack_sessions (list[list[dict]]): List of chat sessions
    """

    print(f"Processing {len(haystack_sessions)} haystack sessions...")

    memory_block = agent_object.memory_block
    can_batch = isinstance(agent_object, ReductiveAgent)    # ReductiveAgent can accept batched user-assistant pairs without its quality being negatively affected

    session_count = 0
    for session in haystack_sessions:
        session_count += 1
        if session_count % 5 == 0:  # Print progress every 5 sessions
            print(f"Processed {session_count}/{len(haystack_sessions)} sessions...")

        msg = None                      # Content from either user or assistance
        pending_user = None             # Temporary storage to ensure user and assistant content is added together
        buffer: List[ChatMessage] = []  # FIFO buffer queue
        batch_pairs = len(session)      # Sets each batch size to the maximum number of pairs in a session (1 LLM-call per session, 500 session per question)

        for turn in session:

            content = turn["content"].replace("<|endoftext|>", "")  # Clean content to avoid tokenizer special-token errors

            # Separate user and assistant messages
            if turn["role"] == "user":
                msg = ChatMessage(role="user", content=content)
            elif turn["role"] == "assistant":
                msg = ChatMessage(role="assistant", content=content)

            if can_batch:
                buffer.append(msg)  # Batch user+assistant messages into buffer and flush when full, reduces LLM-calls

                # Every pair is 2 messages; flush when we reach batch_pairs pairs
                if len(buffer) >= 2 * batch_pairs:
                    try:
                        await memory_block._aput(buffer)    # IMPORTANT: memory_block variable name is constant across agents
                    except Exception as e:
                        print(f"Error processing buffered turns: {e}")
                        traceback.print_exc()
                    buffer = []

            else:   # Ensures only user-assistant pairs are sent
                if msg.role == "user":
                    pending_user = msg
                else:
                    if pending_user is None:
                        continue
                    try:
                        await memory_block._aput([pending_user, msg])   # IMPORTANT: memory_block variable name is constant across agents
                    except Exception as e:
                        print(f"Error processing turn pair: {e}")
                        traceback.print_exc()

                    # Reset temporary variables for next user-assistant pair
                    pending_user = None
                    msg = None

        # End of session: flush any leftover buffered pairs for batched memory blocks
        if can_batch and buffer:
            try:
                await memory_block._aput(buffer)   # IMPORTANT: memory_block variable name is constant across agents
            except Exception as e:
                print(f"Error processing last buffered turns for session: {e}")
                traceback.print_exc()


def reset_memory(agent_object):
    """
    Reset the agent's memory and reinitialize the agent instance.
    Necessary before processing a new question to avoid contamination from previous sessions.

    Args:
        agent_object: The agent instance whose memory is populated
    """

    print(f"Resetting memory...")
    agent_object.memory = agent_object._create_memory()                         # Resets memory
    agent_object.agent = agent_object._create_agent(agent_object.memory, [])    # Resets agent


class LongMemEvalRunner:
    """
    Class to attain a JSON file of questions from the LongMemEval dataset, hypotheses from agents/memoryblocks, and associated results
    Handles token tracking, cost calculation, rate-limit retry, and parallelism.
    """

    def __init__(self, agent):
        self.agent = agent

        # ==============================================================================================================
        # !!! IMPORTANT: CHANGE LIMIT AS NEEDED !!!
        # ==============================================================================================================
        self.semaphore = asyncio.Semaphore(50)   # Limits number of questions processed at a time

    async def process_question(self, question, question_num):
        """
        Evaluate a single question: load history, query the agent, compute token usage & cost.

        Args:
            question (dict): Contains question text, ID, and haystack_sessions
            question_num (int): Index in the current batch

        Returns:
            dict: Results including question_id, hypothesis, tokens, cost, and time, and error (if applicable)
        """

        print(f"Processing question {question_num}: {question.get('question_id', 'unknown')}")

        # Check the structure of the item
        print(f"Item keys: {list(question.keys())}")
        print(f"Number of haystack sessions: {len(question.get('haystack_sessions', []))}")

        # Reset agent memory for this question
        agent_object = create_agent(self.agent)
        print("Memory reset, processing chat history...")

        lch_time = lch_input_tokens = lch_output_tokens = lch_cost = 0          # Initializes variables associated with data around loading the chat history into the memory block
        query_time = query_input_tokens = query_output_tokens = query_cost = 0  # Initializes variables associated with data around the agent's query

        try:
            # Replay all haystack sessions into memory
            await load_chat_history(agent_object, question['haystack_sessions'])

            lch_input_tokens = agent_object.memory_block.input_tokens       # Number of tokens sent to the memory block (haystack_sessions)
            lch_output_tokens = agent_object.memory_block.output_tokens     # Number of tokens returned by the LLM response while parsing haystack_sessions
            lch_time = agent_object.memory_block.load_chat_history_time     # Duration of time the LLM took to parse the haystack_sessions

            lch_cost = (lch_input_tokens / 1000.0) * INPUT_COST_PER_1K + (lch_output_tokens / 1000.0) * OUTPUT_COST_PER_1K  # Calculates real-world cost

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
                except Exception as e:
                    print(f"[DEBUG] Exception type={type(e)} message={e}")
                    if "Rate limit" in str(e) or "429" in str(e):
                        last_exc = e
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        print(f"RateLimitError attempt {attempt}/{RETRY_ATTEMPTS}: {e}. Backing off {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    raise

            if last_exc is not None and answer_text is None:    # Still failing rate-limit after retries
                raise last_exc

            query_input_tokens = agent_object.query_input_tokens    # Number of tokens passed into the agent (Prompt)
            query_output_tokens = agent_object.query_output_tokens  # Number of tokens returned by the agent (Response)
            query_time = agent_object.query_time                    # Duration of time the agent took to respond
            query_cost = (query_input_tokens / 1000.0) * INPUT_COST_PER_1K + (query_output_tokens / 1000.0) * OUTPUT_COST_PER_1K    # Calculates real-world cost

        except Exception as e:
            print(f"Error processing question {question_num + 1}: {e}")
            traceback.print_exc()
            answer_text = f"Error: {str(e)}"

        overall_input_tokens = lch_input_tokens + query_input_tokens    # Total amount of information (prompt/haystack_sessions) passed into the agent/memory pair
        overall_output_tokens = lch_output_tokens + query_output_tokens # Total amount of tokens "returned" by the agent/memory pair
        overall_cost = lch_cost + query_cost                            # Total real-world cost of the question/context
        overall_time = lch_time + query_time                            # Total duration of time to process the question/context

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

        return result

    async def process_question_worker(self, item, i, output_file):
        """
        Worker wrapper: manage semaphore acquisition, run question processing, handle exceptions.

        Args:
            item (dict): Question item
            i (int): Index in dataset
            output_file (str): Path to output file

        Returns:
            dict: Result dictionary including error info if exceptions occur
        """

        acquired = False
        try:
            start = time.monotonic()
            await self.semaphore.acquire()      # No semaphore timeout
            waited = time.monotonic() - start
            print(f"Task {i} acquired semaphore after waiting {waited:.1f}s")
            acquired = True

        except asyncio.TimeoutError:
            print(f"Timeout waiting for semaphore acquisition: {i}")

            # Return an error if semaphore acquisition times out
            result = {
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

            # Save immediately
            await write_results_to_file(output_file, [result], append=True)
            return result

        try:
            # Process the question normally
            result = await self.process_question(item, i)

            # Save immediately after finishing this question
            await write_results_to_file(output_file, [result], append=True)
            return result

        except Exception as e:

            print(f"Unknown error processing question {i}: {e}")

            tb = traceback.format_exc()
            result = {
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

            await write_results_to_file(output_file, [result], append=True)
            return result

        finally:
            # Releases the semaphore only if acquired
            if acquired:
                try:
                    self.semaphore.release()
                    print(f"Released semaphore for task {i}")
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

        # Record starting token counts and time for overall metrics for the entire dataset
        overall_start_time = time.time()

        # Ensure fresh file if starting from zero
        if start_index == 0 and os.path.exists(output_file):
            os.remove(output_file)

        completed_ids = load_completed_ids(output_file)
        print(f"Skipping {len(completed_ids)} already completed questions")
        dataset = [q for q in dataset if q["question_id"] not in completed_ids]

        results = []
        # ==============================================================================================================
        # !!! IMPORTANT: SET BATCH SIZE SAME AS SEMAPHORE LIMIT !!!
        # ==============================================================================================================
        batch_size = 50

        print(f"Running {batch_size} questions in parallel...")

        # Kick off workers in parallel in batches of batch_size
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            batch_results = await asyncio.gather(
                *[self.process_question_worker(item, i, output_file)
                  for i, item in enumerate(batch, start)]
            )
            results.extend(batch_results)

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
            print("CALCULATION DIFFERENCE DETECTED!")

        overall_time = time.time() - overall_start_time

        summary = {
            "memory_loading_prompt_tokens": total_memory_input_tokens,
            "memory_loading_completion_tokens": total_memory_output_tokens,
            "memory_loading_cost": total_memory_cost,
            "query_prompt_tokens": total_query_input_tokens,
            "query_completion_tokens": total_query_output_tokens,
            "query_cost": total_query_cost,
            "total_prompt_tokens": total_combined_input_tokens,
            "total_completion_tokens": total_combined_output_tokens,
            "total_tokens": total_combined_input_tokens + total_combined_output_tokens,
            "total_cost": total_combined_cost,
            "average_prompt_tokens_per_question": mean_combined_input_tokens,
            "average_completion_tokens_per_question": mean_combined_output_tokens,
            "average_total_cost_per_question": mean_combined_cost,
            "total_evaluation_time_seconds": overall_time,
        }

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

        # ==============================================================================================================
        # !!! IMPORTANT: CHANGE SUMMARY FILE NAME AS NEEDED !!!
        # ==============================================================================================================
        summary_file = os.path.join(os.path.dirname(output_file), "reductive_agent_performance_summary.json")

        # --- write summary to JSON file in the same folder as output_file ---
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\nRun summary written to {os.path.abspath(summary_file)}")


def create_agent(agent_class):
    """
    Create a unique instance of the agent.

    Args:
        agent_class: The class of the agent to instantiate

    Returns:
        agent_instance: Instantiated agent
    """

    agent_instance = agent_class()
    return agent_instance


def main():
    """
    Entry point for evaluation script. Configures handler, callback manager,
    agent runner, dataset, and output paths. Runs evaluation asynchronously.
    """

    start_index = 0
    num_questions = 500

    # ==================================================================================================================
    # !!! IMPORTANT: CHANGE AGENT AS NEEDED !!!
    # ==================================================================================================================
    runner = LongMemEvalRunner(ReductiveAgent)

    # Get the directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build the full path to your JSON file
    data_file = os.path.join(BASE_DIR, "eval", "data", "custom_history", "longmemeval_m.json")

    # Example: check it exists before loading
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cannot find dataset file: {data_file}")

    # ==================================================================================================================
    # !!! IMPORTANT: CHANGE FILE NAME BASED ON AGENT !!!
    # ==================================================================================================================
    output_file = "results/reductive_agent_responses.json"

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    # Run evaluation asynchronously
    asyncio.run(
        runner.evaluate_on_dataset(data_file, output_file, start_index=start_index, num_questions=num_questions))


if __name__ == "__main__":
    main()
