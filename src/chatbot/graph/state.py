"""
Defines the shared state object that flows through every node in the graph.
Add new fields here as you add new features — nodes just ignore fields they
don't need, so this is always backward-compatible.
"""

from typing import Annotated, TypedDict, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def replace_summary(old: str, new: str) -> str:
    """Always replace with the latest summary."""
    return new if new is not None else old


class AgentState(TypedDict):
    """
    The central state passed between all LangGraph nodes.

    Fields
    ------
    messages            : Full conversation history (auto-merged by LangGraph)
    messages_summary    : Summary of the messages if message cross certain number of limits.
    """

    messages_summary: Annotated[str, replace_summary]
    messages: Annotated[list[BaseMessage], add_messages]
    


def get_initial_state() -> AgentState:
    """Returns a clean starting state for a new conversation."""
    return AgentState(
        messages_summary="",
        messages=[],
    )
