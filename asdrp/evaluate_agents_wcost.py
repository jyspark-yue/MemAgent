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
#                   (Brought down eval runtime from ~18 mins for 1 question to ~9 mins for 500 questions [Dataset: longmemeval_m.json])
#
# Date:
#   Created:    August 3, 2025  (Varenya Garg)
#   Modified:   August 22, 2025 (Oliver Hsu)
#   Modified:   September 2, 2025 (Eric Vincent Fernandes)
#############################################################################

import json
import asyncio
from asdrp.agent.summary_agent import SummaryAgent
import time
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI
import os
import traceback

# Import the OpenAI RateLimitError to detect rate-limit exceptions
try:
    from openai.error import RateLimitError
except ImportError:
    class RateLimitError(Exception):
        """Fallback RateLimitError used when openai.error isn't importable at analysis time."""
        pass

# To track cost
INPUT_COST_PER_1K  = 0.0003
OUTPUT_COST_PER_1K = 0.0006

RETRY_ATTEMPTS = 3                  # number of retries on rate-limit
RETRY_BASE_DELAY = 1.0              # seconds, exponential backoff base

async def write_results_to_file(output_file, results, append=False):
    mode = 'a' if append else 'w'
    print(f"Saving results to {output_file} (append={append})...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode) as f:
        for result in results:
            # ensure we always write a json object per line
            f.write(json.dumps(result) + "\n")
    print(f"Results saved in {output_file}")

async def load_chat_history(agent_object, haystack_sessions):
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

            if turn["role"] == "user":
                user_msg = ChatMessage(role="user", content=content)
            elif turn["role"] == "assistant":
                agent_text = ChatMessage(role="assistant", content=content)

                if user_msg is not None and agent_text is not None:
                    try:
                        await agent_object.memory_block._aput([user_msg, agent_text]) # memory_block variable is constant!
                    except Exception as e:
                        print(f"Error processing turn: {e}")
                        traceback.print_exc()
                        continue

                    user_msg = None
                    agent_text = None


def reset_memory(agent_object):
    agent_object.memory = agent_object._create_memory()
    agent_object.agent = agent_object._create_agent(agent_object.memory, [])

class LongMemEvalRunner:
    def __init__(self, agent, callback_manager, handler):
        self.agent = agent
        self.callback_manager = callback_manager
        self.handler = handler
        self.semaphore = asyncio.Semaphore(3)

    async def process_question(self, question, question_num):

        print(f"Processing question {question_num}: {question.get('question_id', 'unknown')}")

        # Check the structure of the item
        print(f"Item keys: {list(question.keys())}")
        print(f"Number of haystack sessions: {len(question.get('haystack_sessions', []))}")

        q_t0 = time.time()
        q_prompt0 = self.handler.prompt_llm_token_count
        q_comp0 = self.handler.completion_llm_token_count

        agent_object = create_agent(self.agent, self.callback_manager)
        print("Memory reset, processing chat history...")

        try:
            await load_chat_history(agent_object, question['haystack_sessions'])

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
                # After retries still failing with rate limit
                raise last_exc

        except Exception as e:
            print(f"Error processing question {question_num + 1}: {e}")
            traceback.print_exc()
            answer_text = f"Error: {str(e)}"

        # Compute stats
        q_prompt = self.handler.prompt_llm_token_count - q_prompt0
        q_comp = self.handler.completion_llm_token_count - q_comp0
        q_time = time.time() - q_t0
        q_cost = (q_prompt / 1000.0) * INPUT_COST_PER_1K + (q_comp / 1000.0) * OUTPUT_COST_PER_1K

        print(f"[Q{question_num + 1}] prompt_tokens={q_prompt}  completion_tokens={q_comp}  cost=${q_cost:.6f}  time={q_time:.2f}s")

        result = {
            "question_id": question["question_id"],
            "hypothesis": answer_text,
            "prompt_tokens": q_prompt,
            "completion_tokens": q_comp,
            "cost": q_cost,
            "time": q_time
        }

        if isinstance(answer_text, str) and answer_text.startswith("Error:"):
            result["error"] = answer_text

        return result

    async def process_question_worker(self, item, i):
        """Worker wrapper: acquire semaphore, run question, ensure release, return a dict/result."""

        acquired = False
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=3600)
            acquired = True
        except asyncio.TimeoutError:
            return {
                "question_id": item.get("question_id", f"idx_{i}"),
                "hypothesis": f"Error: timed out waiting for semaphore",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "time": 0.0,
                "error": "semaphore_timeout"
            }

        try:
            result = await self.process_question(item, i)
            return result
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "question_id": item.get("question_id", f"idx_{i}"),
                "hypothesis": f"Error: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "time": 0.0,
                "error": "exception",
                "traceback": tb
            }
        finally:
        # Releases the semaphore only if acquired it
            if acquired:
                try:
                    self.semaphore.release()
                except Exception:
                    pass

    async def evaluate_on_dataset(self, data_file, output_file, num_questions, start_index=0):


        print(f"Loading dataset from {data_file}...")
        with open(data_file, 'r') as f:
            dataset = json.load(f)

        print(f"Dataset loaded with {len(dataset)} total questions")

        if num_questions is not None:
            dataset = dataset[start_index:start_index+num_questions]
            print(f"Processing questions [{start_index}:{start_index + num_questions}] "
              f"(total this run: {len(dataset)})")
        else:
            dataset = dataset[start_index:]
            print(f"Processing questions [{start_index}:] (total this run: {len(dataset)})")

        print(f"Running {len(dataset)} questions in parallel...")

        # To track time
        overall_t0 = time.time()
        overall_prompt0 = self.handler.prompt_llm_token_count
        overall_comp0   = self.handler.completion_llm_token_count

        # Kick off all questions in parallel, semaphore limits to 3 at a time
        tasks = [asyncio.create_task(self.process_question_worker(item, i)) for i, item in enumerate(dataset)]
        results = await asyncio.gather(*tasks)

        append_mode = start_index > 0
        await write_results_to_file(output_file, results, append=append_mode)

        # Calculate cost and time
        overall_prompt = self.handler.prompt_llm_token_count - overall_prompt0
        overall_comp   = self.handler.completion_llm_token_count - overall_comp0
        overall_time   = time.time() - overall_t0
        overall_cost   = (overall_prompt/1000.0)*INPUT_COST_PER_1K + (overall_comp/1000.0)*OUTPUT_COST_PER_1K

        print("\n=== Run summary ===")
        print(f"Total prompt tokens:     {overall_prompt}")
        print(f"Total completion tokens: {overall_comp}")
        print(f"Total tokens:            {overall_prompt + overall_comp}")
        print(f"Estimated total cost:    ${overall_cost:.6f}")
        print(f"Total wall time:         {overall_time:.2f}s")

def create_agent(agent_class, callback_manager=None):
    # Build the LLM with token tracker
    llm = OpenAI(model="gpt-4o",
                 callback_manager=callback_manager)
    agent_instance = agent_class(llm=llm)
    return agent_instance

def main():
    
    # If starting from specific question in dataset
    start_index = 0
    num_questions = 500

    handler = TokenCountingHandler()
    cb_manager = CallbackManager(handlers=[handler])

    runner = LongMemEvalRunner(SummaryAgent, cb_manager, handler)
    
    # Change according to dataset in use
    data_file = "eval/data/custom_history/longmemeval_m.json"

    # Change according to agent analyzed
    output_file = "results/summary_agent_responses.json"

    os.makedirs("results", exist_ok=True)
    
    asyncio.run(runner.evaluate_on_dataset(data_file, output_file, start_index=start_index, num_questions=num_questions))

if __name__ == "__main__":
    main()