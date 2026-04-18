import pytest

llama_types = pytest.importorskip("llama_index.core.base.llms.types")
vm = pytest.importorskip("asdrp.memory.vector_memory")

ChatMessage = llama_types.ChatMessage
VectorMemoryBlock = vm.VectorMemoryBlock


class _FakeNode:
    def __init__(self, text: str) -> None:
        self.node = type("NodePayload", (), {"text": text})()


class _FakeRetriever:
    def __init__(self, docs: list[str], top_k: int) -> None:
        self._docs = docs
        self._top_k = top_k

    def retrieve(self, query: str) -> list[_FakeNode]:
        lowered = query.lower()
        matches = [d for d in self._docs if "kyoto" in d.lower() and "city" in lowered]
        selected = (matches or self._docs)[: self._top_k]
        return [_FakeNode(text) for text in selected]


class _FakeIndex:
    def __init__(self) -> None:
        self.docs: list[str] = []
        self.requested_top_k = 0

    def insert(self, doc) -> None:
        self.docs.append(doc.text)

    def as_retriever(self, similarity_top_k: int = 3) -> _FakeRetriever:
        self.requested_top_k = similarity_top_k
        return _FakeRetriever(self.docs, similarity_top_k)


@pytest.mark.asyncio
async def test_vector_memory_aput_and_aget_smoke(monkeypatch):
    fake_index = _FakeIndex()
    monkeypatch.setattr(
        vm.VectorStoreIndex,
        "from_vector_store",
        staticmethod(lambda **kwargs: fake_index),
    )

    memory = VectorMemoryBlock(
        vector_store=object(),
        embed_model=object(),
        similarity_top_k=2,
    )

    await memory._aput(
        [
            ChatMessage(role="user", content="My favorite city is Kyoto."),
            ChatMessage(role="assistant", content="Thanks, noted."),
        ]
    )

    result = await memory._aget([ChatMessage(role="user", content="What city do I like?")])

    assert "Kyoto" in result
    assert fake_index.requested_top_k == 2
    assert memory.input_tokens > 0
    assert memory.load_chat_history_time >= 0
