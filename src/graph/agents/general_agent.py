import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.prompts import ERROR_MESSAGE
from src.graph.state import State
from src.services.chat_models import get_llm_model
from src.utils.memory_helper import trim_messages_for_llm

logger = logging.getLogger(__name__)


def general_agent_node(state: State) -> dict:
    """Answers greetings, small talk and general questions, without tools."""
    user_message = HumanMessage(content=state["input_message"])
    limited_messages = trim_messages_for_llm(
        [*state["general_agent"], user_message])
    logger.debug("Messages sent to GENERAL AGENT: %s", limited_messages)

    try:
        response = get_llm_model(state["session_id"]).invoke(limited_messages)
    except Exception:
        logger.exception("General agent call failed")
        response = AIMessage(content=ERROR_MESSAGE)

    logger.debug("RESPONSE GENERAL AGENT: %s", response.text)
    return {
        "general_agent": [user_message, response],
        "last_message": response.text,
    }
