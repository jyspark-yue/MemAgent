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
#               - Saved results for each question to JSON file once acquired instead of at the end, safer record keeping
#               - Added functionality to save token metrics to a JSON file after runtime
#
# Date:
#   Created:    August 3, 2025  (Varenya Garg)
#   Modified:   August 22, 2025 (Oliver Hsu)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

import asyncio
import json
import os
import random
import time
import traceback
from typing import List
import uuid

from llama_index.core.base.llms.types import ChatMessage

from aiohttp import ClientConnectorError
from asdrp.agent.summary_agent import SummaryAgent
from asdrp.agent.reductive_agent import ReductiveAgent
from asdrp.agent.episodic_agent import EpisodicAgent
from asdrp.agent.hvm_agent import (
    HVMAgent,
    create_qdrant_client,
    create_default_llm,
    create_default_embedding_model,
)
from google.genai.errors import ClientError

# Import Gemini RateLimitError to detect rate-limit exceptions
try:
    from google.generativeai.errors import RateLimitError
except ImportError:
    # Fallback definition in case google.generativeai package is not available
    class RateLimitError(Exception):
        """Fallback RateLimitError used when generativeai.error isn't importable at analysis time."""

        pass


# Constants to compute approximate cost for API usage (gemini-2.5-flash-lite)
INPUT_COST_PER_1K = (
    0.00010  # Cost per 1000 input tokens ($)    [$0.10 * (1000/1000000)]
)
OUTPUT_COST_PER_1K = (
    0.00040  # Cost per 1000 output tokens ($)   [$0.40 * (1000/1000000)]
)

# Retry configuration for rate-limit handling
RETRY_ATTEMPTS = 8  # Maximum number of retry attempts on errors
RETRY_BASE_DELAY = 1.5  # Base seconds for exponential backoff between retries
RETRY_MAX_DELAY = 20  # Max seconds for exponential backoff between retries
# RETRY_BASE_DELAY = 10  # Base seconds for exponential backoff between retries


async def write_results_to_file(output_file, results, append=False):
    """
    Write evaluation results to a JSONL file (one JSON object per line).
    Creates the output directory if it does not exist.

    Args:
        output_file (str): Path to save results
        results (list[dict]): List of result dictionaries
        append (bool): Append mode if True, overwrite if False
    """

    mode = "a" if append else "w"
    print(f"Saving results to {output_file} (append={append})...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode) as f:
        for result in results:
            f.write(
                json.dumps(result) + "\n"
            )  # ensures there is only one json object per line
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

    print(f"Running {agent_object}...")
    print(f"Processing {len(haystack_sessions)} haystack sessions...")

    memory_block = agent_object.memory_block
    can_batch = (
        isinstance(agent_object, ReductiveAgent)
        or isinstance(agent_object, EpisodicAgent)
        or isinstance(agent_object, HVMAgent)
    )  # ReductiveAgent can accept batched user-assistant pairs without its quality being negatively affected
    batch_all = isinstance(agent_object, HVMAgent)
    all_messages = []  # Used only if batch_all is True

    async def _flush_hvm_batch(force: bool = False):
        """Flush accumulated turns for HVMAgent in manageable chunks."""
        nonlocal all_messages
        # Only HVMAgent uses this path
        if not batch_all:
            return
        while (len(all_messages) >= MAX_HVM_BATCH_MESSAGES) or (
            force and len(all_messages) > 0
        ):
            chunk = all_messages[:MAX_HVM_BATCH_MESSAGES]
            all_messages = all_messages[MAX_HVM_BATCH_MESSAGES:]
            print(
                f"Flushing {len(chunk)} batched HVM turns "
                f"(remaining buffered: {len(all_messages)})..."
            )
            await _retry_aput(memory_block, chunk)
    session_count = 0
    turn_count = 0
    for session in haystack_sessions:
        session_count += 1
        if session_count % 5 == 0:  # Print progress every 5 sessions
            print(f"Processed {session_count}/{len(haystack_sessions)} sessions...")

        msg = None  # Content from either user or assistance
        pending_user = None  # Temporary storage to ensure user and assistant content is added together
        buffer: List[ChatMessage] = []  # FIFO buffer queue
        batch_pairs = len(
            session
        )  # Sets each batch size to the maximum number of pairs in a session (1 LLM-call per session, 500 session per question)

        for turn in session:
            turn_count += 1
            content = turn["content"].replace(
                "<|endoftext|>", ""
            )  # Clean content to avoid tokenizer special-token errors

            # Separate user and assistant messages
            if turn["role"] == "user":
                msg = ChatMessage(role="user", content=content)
            elif turn["role"] == "assistant":
                msg = ChatMessage(role="assistant", content=content)
            else:
                msg = None
            if batch_all:
                if msg is not None:
                    all_messages.append(msg)
                    await _flush_hvm_batch()
            else:
                if can_batch:
                    buffer.append(
                        msg
                    )  # Batch user+assistant messages into buffer and flush when full, reduces LLM-calls

                    # Every pair is 2 messages; flush when we reach batch_pairs pairs
                    if len(buffer) >= 2 * batch_pairs:
                        await _retry_aput(memory_block, buffer)
                        buffer = []

                else:  # Ensures only user-assistant pairs are sent
                    if msg.role == "user":
                        pending_user = msg
                    else:
                        if pending_user is None:
                            continue
                        await _retry_aput(memory_block, [pending_user, msg])

                        # Reset temporary variables for next user-assistant pair
                        pending_user = None
                        msg = None

        # End of session: flush any leftover buffered pairs for batched memory blocks
        if can_batch and buffer and not batch_all:
            await _retry_aput(memory_block, buffer)
    if batch_all:
        await _flush_hvm_batch(force=True)

APUT_TIMEOUT_SECS = 120
MAX_HVM_BATCH_MESSAGES = 250  # limit turns per aput for HVMAgent to avoid long ingests, 


async def _retry_aput(memory_block, buffer, summarize=None):
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            # HARD TIMEOUT around the ingest
            if summarize is None:
                await asyncio.wait_for(memory_block._aput(buffer), timeout=APUT_TIMEOUT_SECS)
            else:
                await asyncio.wait_for(memory_block._aput(buffer, summarize=summarize), timeout=APUT_TIMEOUT_SECS)
            last_exc = None
            break

        except asyncio.TimeoutError as e:
            last_exc = e
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            delay *= random.uniform(0.8, 1.2)
            print(f"[aput] Timeout after {APUT_TIMEOUT_SECS}s (attempt {attempt}/{RETRY_ATTEMPTS}). Backing off {delay:.1f}s...")
            await asyncio.sleep(delay)
            continue

        except (ClientConnectorError, OSError) as e:
            last_exc = e
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            delay *= random.uniform(0.8, 1.2)
            print(f"[aput] Connect error {type(e).__name__} (attempt {attempt}/{RETRY_ATTEMPTS}): {e}. Backing off {delay:.1f}s...")
            await asyncio.sleep(delay)
            continue

        except ClientError as e:
            print(f"[aput] ClientError (attempt {attempt}): {e}")
            last_exc = e
            if getattr(e, "status", None) in [502, 503, 504]:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                delay *= random.uniform(0.8, 1.2)
                print(f"[aput] Transient {e.status}, backoff {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            raise

        except ValueError as e:
            last_exc = e
            if "no candidates" in str(e):
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY + 10)
                delay *= random.uniform(0.8, 1.2)
                print(f"[aput] No candidates, backoff {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            raise

        except Exception as e:
            last_exc = e
            if "Rate limit" in str(e) or "429" in str(e):
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                delay *= random.uniform(0.8, 1.2)
                print(f"[aput] RateLimit, backoff {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            print(f"[aput] Non-retryable error: {e}")
            raise
    if last_exc is not None:
        raise last_exc


def reset_memory(agent_object):
    """
    Reset the agent's memory and reinitialize the agent instance.
    Necessary before processing a new question to avoid contamination from previous sessions.

    Args:
        agent_object: The agent instance whose memory is populated
    """

    print(f"Resetting memory...")
    if isinstance(agent_object, HVMAgent):
        agent_object.reset_session()  # Resets collection
    else:
        agent_object.memory = agent_object._create_memory()  # Resets memory
        agent_object.agent = agent_object._create_agent(
            agent_object.memory, []
        )  # Resets agent

EVAL_BATCH_SIZE = 25  # Number of questions to process in parallel

class LongMemEvalRunner:
    """
    Class to attain a JSON file of questions from the LongMemEval dataset, hypotheses from agents/memoryblocks, and associated results
    Handles token tracking, cost calculation, rate-limit retry, and parallelism.
    """

    def __init__(self, agent):
        self.agent = agent
        if agent is HVMAgent:
            self.q_client = create_qdrant_client()
        else:
            self.q_client = None

        # ==============================================================================================================
        # !!! IMPORTANT: CHANGE LIMIT AS NEEDED !!!
        # ==============================================================================================================
        self.semaphore = asyncio.Semaphore(
            EVAL_BATCH_SIZE
        )  # Limits number of questions processed at a time

    async def process_question(self, question, question_num):
        """
        Evaluate a single question: load history, query the agent, compute token usage & cost.

        Args:
            question (dict): Contains question text, ID, and haystack_sessions
            question_num (int): Index in the current batch

        Returns:
            dict: Results including question_id, hypothesis, tokens, cost, and time, and error (if applicable)
        """

        print(
            f"Processing question {question_num}: {question.get('question_id', 'unknown')}"
        )

        # Check the structure of the item
        print(f"Item keys: {list(question.keys())}")
        print(
            f"Number of haystack sessions: {len(question.get('haystack_sessions', []))}"
        )

        # Reset agent memory for this question
        if self.agent is HVMAgent:
            agent_object = create_agent(self.agent, q_client=self.q_client)
            agent_object.reset_session(question.get("question_id", "unknown"))
        else:
            agent_object = create_agent(self.agent)
        print("Memory reset, processing chat history...")

        lch_time = lch_input_tokens = lch_output_tokens = lch_cost = (
            0  # Initializes variables associated with data around loading the chat history into the memory block
        )
        query_time = query_input_tokens = query_output_tokens = query_cost = (
            0  # Initializes variables associated with data around the agent's query
        )

        try:
            # Replay all haystack sessions into memory
            await load_chat_history(agent_object, question["haystack_sessions"])

            lch_input_tokens = (
                agent_object.memory_block.input_tokens
            )  # Number of tokens sent to the memory block (haystack_sessions)
            lch_output_tokens = (
                agent_object.memory_block.output_tokens
            )  # Number of tokens returned by the LLM response while parsing haystack_sessions
            lch_time = (
                agent_object.memory_block.load_chat_history_time
            )  # Duration of time the LLM took to parse the haystack_sessions

            lch_cost = (lch_input_tokens / 1000.0) * INPUT_COST_PER_1K + (
                lch_output_tokens / 1000.0
            ) * OUTPUT_COST_PER_1K  # Calculates real-world cost

            # Retry mechanism for rate-limit errors
            last_exc = None
            answer_text = None
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    response = await agent_object.achat(question["question"])

                    if (
                        not hasattr(response, "response_str")
                        or not response.response_str
                    ):
                        raise ValueError(
                            "Response has no candidates"
                        )  # Explicitly retry if response empty

                    answer_text = response.response_str
                    print(
                        f"Got response: {answer_text[:100]}..."
                    )  # Show first 100 chars
                    last_exc = None
                    break
                except ValueError as e:
                    if "no candidates" in str(e):
                        delay = min(
                            RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                            RETRY_MAX_DELAY + 10,
                        )
                        delay *= random.uniform(0.8, 1.2)
                        print(f"No candidates, backing off {delay}s before retry...")
                        await asyncio.sleep(delay)
                        last_exc = e
                        continue
                    raise
                except RateLimitError as e:
                    last_exc = e
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY - 5
                    )
                    print(
                        f"RateLimitError attempt {attempt}/{RETRY_ATTEMPTS}: {e}. Backing off {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                except Exception as e:
                    print(f"[DEBUG] Exception type={type(e)} message={e}")
                    raise

            if (
                last_exc is not None and answer_text is None
            ):  # Still failing rate-limit after retries
                raise last_exc

            query_input_tokens = (
                agent_object.query_input_tokens
            )  # Number of tokens passed into the agent (Prompt)
            query_output_tokens = (
                agent_object.query_output_tokens
            )  # Number of tokens returned by the agent (Response)
            query_time = (
                agent_object.query_time
            )  # Duration of time the agent took to respond
            query_cost = (query_input_tokens / 1000.0) * INPUT_COST_PER_1K + (
                query_output_tokens / 1000.0
            ) * OUTPUT_COST_PER_1K  # Calculates real-world cost

        except Exception as e:
            print(f"Error processing question {question_num + 1}: {e}")
            traceback.print_exc()
            answer_text = f"Error: {str(e)}"

        overall_input_tokens = (
            lch_input_tokens + query_input_tokens
        )  # Total amount of information (prompt/haystack_sessions) passed into the agent/memory pair
        overall_output_tokens = (
            lch_output_tokens + query_output_tokens
        )  # Total amount of tokens "returned" by the agent/memory pair
        overall_cost = (
            lch_cost + query_cost
        )  # Total real-world cost of the question/context
        overall_time = (
            lch_time + query_time
        )  # Total duration of time to process the question/context

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
            "question_time": overall_time,
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
            await self.semaphore.acquire()  # No semaphore timeout
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
                "error": "semaphore_timeout",
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
                "traceback": tb,
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

    async def evaluate_on_dataset(
        self,
        data_file,
        output_file,
        summary_file,
        num_questions,
        start_index=0,
    ):
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
        with open(data_file, "r") as f:
            dataset = json.load(f)

        print(f"Dataset loaded with {len(dataset)} total questions")

        # Slice dataset for partial evaluation runs
        if num_questions is not None:
            dataset = dataset[start_index : start_index + num_questions]
            print(
                f"Processing questions [{start_index}:{start_index + num_questions}] "
                f"(total this run: {len(dataset)})"
            )
        else:
            dataset = dataset[start_index:]
            print(
                f"Processing questions [{start_index}:] (total this run: {len(dataset)})"
            )

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
        batch_size = EVAL_BATCH_SIZE

        print(f"Running {batch_size} questions in parallel...")

        # Kick off workers in parallel in batches of batch_size
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            batch_results = await asyncio.gather(
                *[
                    self.process_question_worker(item, i, output_file)
                    for i, item in enumerate(batch, start)
                ]
            )
            results.extend(batch_results)

        total_memory_input_tokens = sum(
            r.get("memory_input_tokens", 0) for r in results
        )
        total_memory_output_tokens = sum(
            r.get("memory_output_tokens", 0) for r in results
        )
        total_memory_cost = sum(r.get("memory_cost", 0.00) for r in results)

        total_query_input_tokens = sum(r.get("query_input_tokens", 0) for r in results)
        total_query_output_tokens = sum(
            r.get("query_output_tokens", 0) for r in results
        )
        total_query_cost = sum(r.get("query_cost", 0.00) for r in results)

        total_combined_input_tokens = sum(
            r.get("question_input_tokens", 0) for r in results
        )
        total_combined_output_tokens = sum(
            r.get("question_output_tokens", 0) for r in results
        )
        total_combined_cost = sum(r.get("question_cost", 0) for r in results)

        mean_combined_input_tokens = total_combined_input_tokens / len(dataset)
        mean_combined_output_tokens = total_combined_output_tokens / len(dataset)
        mean_combined_cost = total_combined_cost / len(dataset)

        if total_combined_input_tokens != (
            total_memory_input_tokens + total_query_input_tokens
        ):
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
        print(
            f"Total tokens:                           {total_combined_input_tokens + total_combined_output_tokens}"
        )
        print(f"Total cost:                             ${total_combined_cost:.6f}")
        print(f"Average prompt tokens per question:     {mean_combined_input_tokens}")
        print(f"Average completion tokens per question: {mean_combined_output_tokens}")
        print(f"Average total cost per question:        ${mean_combined_cost:.6f}")
        print(f"Total evaluation time:                  {overall_time:.4f}s")

        # ==============================================================================================================
        # !!! IMPORTANT: CHANGE SUMMARY FILE NAME AS NEEDED !!!
        # ==============================================================================================================
        summary_file = os.path.join(os.path.dirname(output_file), summary_file)

        # --- write summary to JSON file in the same folder as output_file ---
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\nRun summary written to {os.path.abspath(summary_file)}")


def create_agent(agent_class, **kwargs):
    """
    Create a unique instance of the agent.

    Args:
        agent_class: The class of the agent to instantiate

    Returns:
        agent_instance: Instantiated agent
    """
    agent_instance = agent_class(**kwargs)
    return agent_instance


def main(data_file, output_file, summary_file):
    """
    Entry point for evaluation script. Configures handler, callback manager,
    agent runner, dataset, and output paths. Runs evaluation asynchronously.
    """

    start_index = 0
    num_questions = 500

    # ==================================================================================================================
    # !!! IMPORTANT: CHANGE AGENT AS NEEDED !!!
    # ==================================================================================================================
    # runner = LongMemEvalRunner(EpisodicAgent)
    runner = LongMemEvalRunner(HVMAgent)

    # Example: check it exists before loading
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cannot find dataset file: {data_file}")

    # ==================================================================================================================
    # !!! IMPORTANT: CHANGE FILE NAME BASED ON AGENT !!!
    # ==================================================================================================================
    output_file = f"asdrp/results/{output_file}"

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    # Run evaluation asynchronously
    asyncio.run(
        runner.evaluate_on_dataset(
            data_file,
            output_file,
            summary_file=summary_file,
            start_index=start_index,
            num_questions=num_questions,
        )
    )


if __name__ == "__main__":
    # Get the directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build the full path to your JSON file
    # data_file = os.path.join(
    #     BASE_DIR, "eval", "data", "custom_history", "longmemeval_m.json"
    # )
    # data_file = os.path.join(
    #     BASE_DIR,
    #     "eval",
    #     "data",
    #     "custom_history",
    #     "longmemeval_m_500_500_splite_10files",
    #     "longmemeval_m_cleaned_part_02.json",
    # )
    test_files = [
        "longmemeval_m_cleaned_part_01.json",
        "longmemeval_m_cleaned_part_02.json",
        "longmemeval_m_cleaned_part_03.json",
        "longmemeval_m_cleaned_part_04.json",
        "longmemeval_m_cleaned_part_05.json",
        "longmemeval_m_cleaned_part_06.json",
        "longmemeval_m_cleaned_part_07.json",
        "longmemeval_m_cleaned_part_08.json",
        "longmemeval_m_cleaned_part_09.json",
        "longmemeval_m_cleaned_part_10.json",
    ]
    # for f in test_files:
    #     data_file = os.path.join(
    #         BASE_DIR,
    #         "eval",
    #         "data",
    #         "custom_history",
    #         "longmemeval_m_500_500_splite_10files",
    #         f,
    #     )
    #     print(f"Running evaluation on {data_file}...")
    #     output_file = f"hvm_agent_responses_{f.replace('.json', '')}.json"
    #     summary_file = f"hvm_agent_summary_{f.replace('.json', '')}.json"
    #     print(f"Output file: {output_file}, Summary file: {summary_file}")
    #     main(
    #         data_file,
    #         output_file=output_file,
    #         summary_file=summary_file,
    #     )
    data_file = os.path.join(
        BASE_DIR,
        "eval",
        "data",
        "custom_history",
        "longmemeval_m_sample5_20.json",
    )
    main(
        data_file,
        output_file="hvm_agent_responses_m_cleaned_part_01.json",
        summary_file="hvm_agent_summary_m_cleaned_part_01.json",
    )
