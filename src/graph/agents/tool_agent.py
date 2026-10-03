import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph.prompts import ERROR_MESSAGE
from src.graph.state import State
from src.services.chat_models import get_llm_model
from src.tools import ALL_TOOLS
from src.utils.memory_helper import trim_messages_for_llm

logger = logging.getLogger(__name__)


def tool_agent_node(state: State) -> dict:
    """
    Answers using tools. Runs in a loop with the ToolNode (see the tool
    subgraph) until the model answers without calling a tool (Standard LangGraph behavior).
    """
    history = state["tool_agent"]
    new_messages = []
    # The first call of a turn adds the user's message.
    if not history or not isinstance(history[-1], ToolMessage):
        new_messages.append(HumanMessage(content=state["input_message"]))

    limited_messages = trim_messages_for_llm([*history, *new_messages])
    logger.debug("Messages sent to TOOL AGENT: %s", limited_messages)

    try:
        response = get_llm_model(state["session_id"]).bind_tools(
            ALL_TOOLS).invoke(limited_messages)
    except Exception:
        logger.exception("Tool agent call failed")
        response = AIMessage(content=ERROR_MESSAGE)

    logger.debug("RESPONSE TOOL AGENT: %s | tool calls: %s",
                 response.text, getattr(response, "tool_calls", []))
    new_messages.append(response)
    return {
        "tool_agent": new_messages,
        "last_message": response.text,
    }
