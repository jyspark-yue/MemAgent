#############################################################################
# File: base.py
#
# Description:
#   Defines the small AgentReply record used by the legacy agent interface.
#
#   - Stores the final response text in one dataclass field.
#   - Remains for compatibility with code that expects AgentReply.
#############################################################################


from dataclasses import dataclass


@dataclass
class AgentReply:
    response_str: str
