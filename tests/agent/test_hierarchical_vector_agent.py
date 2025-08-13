"""Tree‑building smoke‑test (does not call OpenAI)

After >4 inserts (default branching factor), TreeIndex creates parent summaries
and  ``print_memory_tree`` will show a real hierarchy.

Run from repo root:
    python -m tests.agent.hvm_tree_demo
"""

from asdrp.agent.hierarchical_vector_agent import HVMAgent

def main() -> None:
    agent = HVMAgent()

    # Conversation
    turns = [
        "USER: Hi, I'm Judy. Remember that I love AI.",
        "ASSISTANT: Noted! You love AI.",
        "USER: My favourite language is Python.",
        "ASSISTANT: Got it – Python is your favourite.",
        "USER: I also enjoy studying large language models.",
        "ASSISTANT: I'll keep that in mind as well.",
    ]

    for t in turns:
        agent.mem.store(t)

    # Printing the memory tree / roots if no tree
    print("\n=== MEMORY TREE ===")
    agent.print_memory_tree()

    # Printing the output for retrieval
    query = "What programming language does Judy like?"
    answer = agent.mem.retrieve(query)
    print("\n=== RETRIEVAL FOR: \"" + query + "\" ===")
    for i, snippet in enumerate(answer, 1):
        print(f"{i}. {snippet}")


if __name__ == "__main__":
    main()
