import logging

from langchain_core.messages import HumanMessage

from src.graph.guardrails.guardrail_helper import (
    guardrail_output_message,
    run_guardrail,
)
from src.graph.state import State

logger = logging.getLogger(__name__)


def input_guardrail_node(state: State) -> dict:
    """Checks the user's message before routing to agents."""
    user_message = HumanMessage(content=state["input_message"])
    guardrail_output = run_guardrail(
        state["input_guardrail_agent"], user_message, state["session_id"])
    logger.info("INPUT GUARDRAIL -> %s %s",
                guardrail_output.result, guardrail_output.reason)

    return {
        "input_guardrail_agent": [user_message, guardrail_output_message(guardrail_output)],
        "input_guardrail_result": guardrail_output.model_dump(),
    }
