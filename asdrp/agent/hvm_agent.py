#############################################################################
# hvm_agent.py
#
# agent that uses HierarchicalVectorMemory block with qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.coom
#############################################################################


from __future__ import annotations

import asyncio
import os
from typing import List, Optional
import uuid
import time

from qdrant_client import QdrantClient, AsyncQdrantClient
from asdrp.agent.base import AgentReply
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from asdrp.memory.hvm import (  # keep your existing module path
    HierarchicalVectorMemory,
    QueryModes,
    create_hvm,
    create_default_llm,
    create_default_embedding_model,
    data_loader,
    create_qdrant_client,
)
from google.genai import types
from dotenv import load_dotenv, find_dotenv
from llama_index.core.base.llms.types import ChatMessage
from asdrp.agent.base import AgentBase

load_dotenv(find_dotenv())


def get_cost_per_1k_tokens(
    provider: str, model: str, token_type: str = "input"
) -> float:
    """
    Get cost per 1000 tokens for a given provider and model.

    Args:
        provider: LLM provider (only "gemini" supported in this version)
        model: Model name
        token_type: "input" or "output"

    Returns:
        Cost per 1000 tokens in USD
    """

    # Gemini pricing (as of 2024)
    gemini_pricing = {
        "models/gemini-2.5-flash-lite": {
            "input": 0.000075,
            "output": 0.0003,
        },  # 使用与1.5-flash相同的定价
        "models/gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "models/gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "models/gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        "models/embedding-001": {"input": 0.000025, "output": 0.0},
    }

    if provider.lower() == "gemini":
        pricing = gemini_pricing.get(
            model, {"input": 0.000075, "output": 0.0003}
        )  # Default fallback
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Only 'gemini' is supported in this version."
        )

    return pricing.get(token_type, 0.0)


class HVMAgent(AgentBase):
    """Conversational agent that uses hierarchical vector memory."""

    @property
    def can_batch(self):
        return True
    
    @property
    def batch_all(self):
        return True

    def __init__(
        self,
        q_client: AsyncQdrantClient,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,  # **CHANGED** (type fix)
        top_k: int = 2,
        retrieval_mode: QueryModes = QueryModes.tree_traversal,  # **CHANGED** (typo fix)
        tools: Optional[List[FunctionTool]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.tools = tools or []

        # Configuration so memory reset can rebuild identical state
        self.llm = llm or create_default_llm()  # uses shared Token handler in hvm.py
        self.embed_model = embed_model or create_default_embedding_model()
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        # Vector store / persistence configuration
        self.collection = session_id or f"agent_mem_hvm_{uuid.uuid4().hex}"
        self.q_client = q_client

        # Memory storage / state
        self.memory_block_map = {}

        self.agent = FunctionAgent(
            llm=self.llm,
            tools=tools,
        )

        # Token tracking for cost calculation
        import tiktoken

        self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
        self.embeding_tokens = 0
        self.query_input_tokens = (
            0  # Number of tokens passed into the LLM within this agent
        )
        self.query_output_tokens = (
            0  # Number of tokens returned by the LLM within this agent
        )
        self.query_time = 0  # Duration of time the LLM took to respond

    def reset_session(self, new_session_id: Optional[str] = None) -> None:
        """Reset the agent's session, optionally with a new session ID."""
        self.collection = new_session_id or f"agent_mem_hvm_{uuid.uuid4().hex}"
        # Optionally reset per-question counters here if desired:
        # self.query_input_tokens = 0
        # self.query_output_tokens = 0
        # self.query_time = 0

    @property
    def memory_block(self) -> HierarchicalVectorMemory:
        return self.memory

    @property
    def memory(self) -> HierarchicalVectorMemory:
        # here memory is the memory block since memory has some issue of retrieve info
        if self.collection not in self.memory_block_map:
            self.memory_block_map[self.collection] = create_hvm(
                name="hvm_memory",
                collection_name=self.collection,
                client=self.q_client,
                tree_depth=3,
                llm=self.llm,
                embed_model=self.embed_model,
                transformations=None,
            )
        return self.memory_block_map[self.collection]

    async def achat(self, user_msg: str) -> AgentReply:

        chat_history = await self.memory.aget(
            [ChatMessage(role="user", content=user_msg)],
            mode=self.retrieval_mode,
            similarity_top_k=self.top_k,
        )
        # print(f"Chat history retrieved from HVM: {chat_history}")

        prompt = (
            "You are a helpful, concise assistant.\n\n"
            f"Given the following known context (may be empty), answer the user's question:\n{chat_history}\n\n"
            f"User: {user_msg}\n"
            "Assistant:"
        )

        # ------------------------------
        # **CHANGED**: measure query time
        # ------------------------------
        initial_query_time = time.time()  # **CHANGED**

        # --------------------------------------------------------------------
        # **CHANGED**: Snapshot TokenCountingHandler totals BEFORE the LLM call
        # --------------------------------------------------------------------
        handler = None  # **CHANGED**
        try:
            handlers = getattr(self.llm.callback_manager, "handlers", []) or []  # **CHANGED**
            handler = next(
                (h for h in handlers if isinstance(h, TokenCountingHandler)), None  # **CHANGED**
            )
        except Exception:
            handler = None  # **CHANGED**
        p0 = handler.prompt_llm_token_count if handler else 0  # **CHANGED**
        c0 = handler.completion_llm_token_count if handler else 0  # **CHANGED**

        # Ask the LLM for a completion (native async)
        print(f"Prompt to LLM:\n{prompt}\n")
        assistant_msg = await self.agent.run(prompt)
        assistant_msg = (
            assistant_msg.response.content
            if isinstance(assistant_msg, ChatMessage)
            else str(assistant_msg)
        )

        # -------------------------------------------------------------------
        # **CHANGED**: Compute token deltas (prefer handler; fallback tokenizer)
        # -------------------------------------------------------------------
        if handler:  # **CHANGED**
            self.query_input_tokens += max(0, handler.prompt_llm_token_count - p0)   # **CHANGED**
            self.query_output_tokens += max(0, handler.completion_llm_token_count - c0)  # **CHANGED**
        else:  # **CHANGED**
            self.query_input_tokens += len(self.tokenizer.encode(prompt))  # **CHANGED**
            self.query_output_tokens += len(self.tokenizer.encode(assistant_msg))  # **CHANGED**

        # ------------------------------
        # **CHANGED**: finalize query time
        # ------------------------------
        self.query_time = time.time() - initial_query_time  # **CHANGED**

        # Store the turn to memory (this will count into memory-block ingest tokens)
        await self.memory.aput(
            [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=assistant_msg),
            ]
        )

        _output = AgentReply(response_str=assistant_msg)
        print(f"#####session_id:{self.collection}, output:{_output.response_str}")
        return _output


async def run_smoke_test():
    # Example usage of data_loader
    try:
        # Try to load the default dataset file
        # dataset = data_loader("longmemeval_m_sample5_20.json")
        # dataset = data_loader("longmemeval_single_500.json")
        dataset = data_loader(
            "/Users/judyyu/memagents/asdrp/eval/data/custom_history/longmemeval_m_sample5_20.json"
        )
        print(f"Successfully loaded dataset with {len(dataset)} items")

    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        raise e
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    print("Running smoke test...")
    print(f"Dataset contains {len(dataset)} items.")
    base_session_id = "test_hvm_session" + str(uuid.uuid4().hex)
    q_client = create_qdrant_client()

    # --------------------------------------------------------------------
    # **CHANGED**: use create_default_llm() without overriding callback_manager
    # so it shares the global TokenCountingHandler with embeddings (from hvm)
    # --------------------------------------------------------------------
    hvm_agent = HVMAgent(
        q_client=q_client,
        llm=create_default_llm(),                    # **CHANGED** (remove CallbackManager([]))
        embed_model=create_default_embedding_model(),# **CHANGED**
        top_k=1,
        retrieval_mode=QueryModes.tree_traversal,
        tools=[],
        session_id=base_session_id,
    )
    try:
        for q in dataset:

            print("--------------------------------")
            print("+++++++++++++++++++++++++++++++++++")
            print("--------------------------------")
            haystack_sessions = q.get("haystack_sessions", [])
            haystack_session_ids = q.get("haystack_session_ids")
            question_id = q.get("question_id")
            question = q.get("question")
            answer = q.get("answer")
            answer_session_ids = q.get("answer_session_ids", [])
            hvm_agent.reset_session(
                base_session_id + "_" + question_id
            )  # Reset session for each new question
            print(
                f"Processing question with {len(haystack_sessions)} haystack sessions."
            )

            total_turn = 0
            for idx, session in enumerate(haystack_sessions):
                print(f"Processing haystack session {idx} with {len(session)} turns.")
                buffer = []
                for turn in session:
                    content = turn["content"].replace(
                        "<|endoftext|>", ""
                    )  # Clean content to avoid tokenizer special-token errors

                    if turn["role"] == "user":
                        msg = ChatMessage(role="user", content=content)
                    elif turn["role"] == "assistant":
                        msg = ChatMessage(role="assistant", content=content)
                    else:
                        raise ValueError(f"Unknown role: {turn['role']}")
                    buffer.append(msg)
                await hvm_agent.memory.aput(buffer)
                total_turn += len(buffer)
            print(
                f"Inserted all haystack sessions into HVM. Total turns inserted for this question: {total_turn}"
            )

            response = await hvm_agent.achat(question)
            print(f"Question: {question}")
            print(f"Retrieved context from HVM.\n{response}")
            print(f"Answer: {answer}")

            # Optional: print per-question agent token/time counters
            print(
                f"[Agent Query Stats] prompt={hvm_agent.query_input_tokens}, "
                f"completion={hvm_agent.query_output_tokens}, time={hvm_agent.query_time:.3f}s"
            )
            print(
                f"[Memory Load Stats] prompt={hvm_agent.memory.input_tokens}, "
                f"completion={hvm_agent.memory.output_tokens}, "
                f"embed={hvm_agent.memory.embed_tokens}, "
                f"load_time={hvm_agent.memory.load_chat_history_time:.3f}s"
            )

    finally:
        print("Closing HVM...")


if __name__ == "__main__":
    time_start = time.time()
    asyncio.run(run_smoke_test())
    print(f"Total time: {time.time() - time_start} seconds")
