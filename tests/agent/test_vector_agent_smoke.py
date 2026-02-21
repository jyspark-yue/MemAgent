from types import SimpleNamespace

import pytest

va = pytest.importorskip("asdrp.agent.vector_agent")

VectorAgent = va.VectorAgent


class _FakeLLM:
    async def acomplete(self, prompt: str):
        return SimpleNamespace(text="You like Kyoto.")


class _FakeVectorMemoryBlock:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def _aget(self, messages):
        return "User preference: favorite city is Kyoto."

    async def _aput(self, messages):
        return None


@pytest.mark.asyncio
async def test_vector_agent_achat_smoke(monkeypatch):
    monkeypatch.setattr(va, "get_default_llm", lambda: _FakeLLM())
    monkeypatch.setattr(va, "QdrantClient", lambda **kwargs: object())
    monkeypatch.setattr(va, "AsyncQdrantClient", lambda **kwargs: object())
    monkeypatch.setattr(va, "QdrantVectorStore", lambda **kwargs: object())
    monkeypatch.setattr(va, "GeminiEmbedding", lambda **kwargs: object())
    monkeypatch.setattr(va, "VectorMemoryBlock", _FakeVectorMemoryBlock)

    agent = VectorAgent(collection="smoke_vector_mem")
    reply = await agent.achat("What city do I like?")

    assert "Kyoto" in reply.response_str
    assert agent.query_input_tokens > 0
    assert agent.query_output_tokens > 0
    assert agent.query_time >= 0
