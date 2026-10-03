import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import State
from src.services.chat_models import get_llm_router_model
from src.utils.memory_helper import trim_messages_for_llm

logger = logging.getLogger(__name__)

# Categories the classifier can return. The model can only answer one of these.
# Keep in sync with get_input_classifier_prompt() in src/graph/prompts.py
Category = Literal["general", "tools"]

# Agent that answers each category. Every Category needs an entry here.
CATEGORY_TO_AGENT: dict[Category, str] = {
    "general": "general_agent",
    "tools": "tool_agent",
}
# Only used when the classifier call fails
DEFAULT_CATEGORY: Category = "general"


class InputClassification(BaseModel):
    """Category of the user's latest message."""

    category: Category = Field(
        description="The category that best matches the user's latest message.")


def input_classifier_node(state: State) -> dict:
    """Asks the router model which agent should answer, and saves it in `route`."""
    user_message = HumanMessage(content=state["input_message"])
    llm_router_structured = get_llm_router_model(
        state["session_id"]).with_structured_output(InputClassification, method="json_schema")

    try:
        output = llm_router_structured.invoke(
            trim_messages_for_llm([*state["input_classifier"], user_message]))
    except Exception:
        logger.exception("Input classifier call failed")
        output = None

    category = output.category if isinstance(output, InputClassification) else DEFAULT_CATEGORY
    route = CATEGORY_TO_AGENT[category]
    logger.info("INPUT CLASSIFIER -> %r, routing to %s", category, route)

    return {
        # Saving both results so the router agent can route follow-up questions more easily
        "input_classifier": [user_message, AIMessage(content=category)],
        "route": route
    }
