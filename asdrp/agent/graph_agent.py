
#############################################################################
# graph_agent.py
# 
# Graph Based Memory Agent using Mem0 architecture
# 
# Setup: mem0ai llama-index  python-dotenv
#
# @author Ayden Grover
# @version 27 September 2025
#############################################################################


import os
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- LLM (LlamaIndex OpenAI wrapper) ---
from llama_index.llms.openai import OpenAI



# Lightweight AgentReply struct
# =============================
@dataclass
class AgentReply:
    response_str: str

# Mem0 backend adapter (w/fallback)
try:
    from mem0 import MemoryClient  # type: ignore
    _HAS_MEM0 = True
except Exception:
    MemoryClient = None  # type: ignore
    _HAS_MEM0 = False

class Mem0Backend:
    """Thin adapter for Mem0's graph(+vector) memory.

    Provides two methods:
      - add(user_id, text) -> str
      - search(user_id, query, k=5) -> List[{"id","text","score"}]

    If Mem0 is not available, falls back to a simple in-memory stub so the
    module remains runnable.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._is_stub = False
        if _HAS_MEM0:
            cfg = config or self._config_from_env()
            self._client = MemoryClient(config=cfg)
        else:
            self._is_stub = True
            self._store: Dict[str, List[Tuple[str, str]]] = {}
            self._id_counter = 0

    @staticmethod
    def _config_from_env() -> Dict[str, Any]:
        graph_provider = os.getenv("MEM0_GRAPH_PROVIDER", "neo4j")
        vector_provider = os.getenv("MEM0_VECTOR_PROVIDER")
        cfg: Dict[str, Any] = {
            "store": {
                "graph": {
                    "provider": graph_provider,
                    "uri": os.getenv("MEM0_GRAPH_URI"),
                    "user": os.getenv("MEM0_GRAPH_USER"),
                    "password": os.getenv("MEM0_GRAPH_PASSWORD"),
                }
            },
            "llm": {
                "provider": os.getenv("MEM0_LLM_PROVIDER", "openai"),
                "model": os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"),
            },
        }
        if vector_provider:
            cfg["store"]["vector"] = {"provider": vector_provider}
        return cfg

    def add(self, user_id: str, text: str) -> str:
        if not text:
            return ""
        if self._is_stub:
            self._id_counter += 1
            mem_id = f"mem-{self._id_counter}"
            self._store.setdefault(user_id, []).append((mem_id, text))
            return mem_id
        else:
            mem_id = self._client.add(user_id, text)
            if isinstance(mem_id, list):
                return mem_id[0] if mem_id else ""
            return str(mem_id)

    def search(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._is_stub:
            items = self._store.get(user_id, [])
            q = query.lower()
            scored = []
            for mem_id, txt in items:
                score = 1.0 if q in txt.lower() else 0.0
                scored.append({"id": mem_id, "text": txt, "score": score})
            scored.sort(key=lambda r: r["score"], reverse=True)
            return scored[:k]
        else:
            results = self._client.search(user_id, query)
            out: List[Dict[str, Any]] = []
            for r in results[:k]:
                if isinstance(r, dict):
                    mem_id = r.get("id") or r.get("_id") or ""
                    text = r.get("text") or r.get("content") or r.get("value") or ""
                    score = r.get("score") or r.get("similarity") or 0.0
                    out.append({"id": str(mem_id), "text": str(text), "score": float(score) if score else 0.0})
                else:
                    out.append({"id": "", "text": str(r), "score": 0.0})
            return out

# Minimal Mem0 agent
class Mem0Agent:
    """
      retrieve → prompt → answer → store
    """
    def __init__(self, model: str = "gpt-4o-mini", top_k: int = 3, user_id: str = "user-123") -> None:
        self.llm = OpenAI(model=model)
        self.backend = Mem0Backend()
        self.top_k = top_k
        self.user_id = user_id
        #self.tokenizer = tiktoken.get_encoding("o200k_base")

    async def achat(self, user_msg: str) -> AgentReply:
        try:
            # Retrieve relevant snippets from Mem0
            hits = self.backend.search(self.user_id, user_msg, k=self.top_k)
            context = "\n".join(h.get("text", "") for h in hits if h.get("text"))

            # Compose prompt similar to HVMAgent style
            prompt = (
                "You are a helpful, concise assistant.\n\n"
                f"Known context (may be empty):\n{context}\n\n"
                f"User: {user_msg}\n"
                "Assistant:"
            )

            # Ask the LLM (single-turn completion)
            completion = self.llm.complete(prompt)
            assistant_msg: str = completion.text.strip()

            # Store the turn in Mem0 (Mem0 will extract entities/relations)
            self.backend.add(self.user_id, f"USER: {user_msg}")
            self.backend.add(self.user_id, f"ASSISTANT: {assistant_msg}")

            return AgentReply(response_str=assistant_msg)
        except Exception as e:
            return AgentReply(response_str=f"Error in Mem0Agent: {e}")

# =====================
# Smoke test runner
# =====================
async def _smoke_tests():
    agent = Mem0Agent()

    print("Test 1: Basic QA with retrieval")
    _ = await agent.achat("I moved to Paris in March.")
    ans = await agent.achat("Which city did I move to?")
    print(ans.response_str)

    print("\nTest 2: Multi-fact reference")
    _ = await agent.achat("The sky is blue.")
    _ = await agent.achat("Water is wet.")
    ans2 = await agent.achat("What do we know about the sky and water?")
    print(ans2.response_str)

if __name__ == "__main__":
    asyncio.run(_smoke_tests())
