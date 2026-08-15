#############################################################################
# File: __init__.py
#
# Description:
#   Exposes the active memory agents through one package entry point. The
#   evaluator uses this registry to select an architecture from its
#   command-line name.
#
#   - Imports each supported agent implementation once.
#   - Maps vector, episodic, condensed, proposition, HVM, graph, and
#     Qdrant graph names to their classes.
#   - Lists the package's public agent exports.
#############################################################################

from asdrp.agent.episodic_agent import EpisodicAgent
from asdrp.agent.graph_agent import GraphAgent
from asdrp.agent.hvm_agent import HVMAgent
from asdrp.agent.reductive_agent import ReductiveAgent
from asdrp.agent.summary_agent import SummaryAgent
from asdrp.agent.vector_agent import VectorAgent

# Map each CLI value to the agent class it should build.
AGENT_REGISTRY = {
    "vector": VectorAgent,  # Raw vector baseline.
    "episodic": EpisodicAgent,  # Event memory.
    "condensed": SummaryAgent,  # Recursive summary memory.
    "proposition": ReductiveAgent,  # Atomic fact memory.
    "hvm": HVMAgent,  # RAPTOR-style hierarchy.
    "graph": GraphAgent,  # In-memory graph traversal.
}

# Keep the public imports clear for tools that inspect this package.
__all__ = [
    "AGENT_REGISTRY",
    "EpisodicAgent",
    "GraphAgent",
    "HVMAgent",
    "ReductiveAgent",
    "SummaryAgent",
    "VectorAgent",
]
