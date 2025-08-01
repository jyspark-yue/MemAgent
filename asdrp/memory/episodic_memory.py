#############################################################################
#   File:   episodic_memory.py
#   @author Eric Vincent Fernandes
#   @email  evfdes@gmail.com
#   @date   Jul 31, 2025
#
#   Extracts episodic data [timestamp, user input, agent output, location,
#       outcome (was agent response accepted), reflection (learnings from interaction), and categorical tags from interaction]
#############################################################################

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.node_parser import SimpleNodeParser


class EpisodicMemoryBlock(BaseMemoryBlock[str]):

    def __init__(self):
        super().__init__()
        self.memories: List[Document] = []
        self.index = None

    def add_episode(
            self,
            user_input: str,
            agent_output: str,
            location: Optional[str] = None,                 #   Location specified in interaction
            outcome: Optional[str] = None,                  #   Success of the agent's output
            reflection: Optional[str] = None,               #   Key learnings from interaction
            categorical_tags: Optional[List[str]] = None,   #   Key tags to categorize the interaction
    ) -> None:
        """Adds a new entry (episode/interaction/memory) into memory."""

        episode_id = str(uuid4())                                   #   Generates a unique identifier per episode
        episode_timestamp = datetime.now(timezone.utc).isoformat()   #   Records timestamp with timezone
        categorical_tags = categorical_tags or []                   #   Creates an empty list (separated from initialization to prevent mutability issues)

        episode_text = f"""
            [USER INPUT]: {user_input}
            [AGENT OUTPUT]: {agent_output}
            [OUTCOME]: {outcome}
            [LOCATION]: {location}
            [REFLECTION]: {reflection}
        """         #   Saves key parts of interaction into memory

        metadata = {
            "id": episode_id,
            "timestamp": episode_timestamp,
            "user_input": user_input,
            "agent_output": agent_output,
            "location": location,
            "outcome": outcome,
            "reflection": reflection,
            "categorical_tags": categorical_tags,
        }           #   For use when searching through different memories

        episode = Document(text = episode_text.strip(), metadata = metadata)
        self.memories.append(episode)

        if self.index is None:
            parser = SimpleNodeParser()                         #   Breaks down large memories (Documents) into smaller chunks (nodes)
            nodes = parser.get_nodes_from_memory(episode)       #   Converts document objects into nodes
            self.index = VectorStoreIndex(nodes)                #   Vectorizes nodes and stores in VectorStore (LlamaIndex)
        else:
            self.index.insert(episode)                          #   Converts, vectorizes, and adds a new document into the VectorStore

    def get_relevant_memories(
            self,
            query: str,         #   Requested information
            top_x: int = 5      #   The top X most relevant memories to the query
    ) -> List[str]:
        """Returns a list of relevant entries (episodes/interactions/memories) based on a given query."""

        if self.index is None:
            return ["No Memories Stored"]

        query_engine = self.index.as_query_engine(max_num_relevant = top_x) #   Configures response synthesizer for top X entries
        response = query_engine.query(query)                                #   Finds top X most relevant entries
        return [node.getContent() for node in response.source_nodes]        #   Extracts and returns content in relevant memory entries

    def get_all_memories(self) -> List[str]:
        """Returns a list of all entries (episodes/interactions/memories) stored in memory."""
        return [doc.text for doc in self.memories]

    def reset_memories(self) -> None:
        """Removes all entries (episodes/interactions/memories) stored in memory."""
        self.memories = []
        self.index = None