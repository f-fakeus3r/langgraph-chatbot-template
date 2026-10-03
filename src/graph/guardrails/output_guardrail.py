import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph import prompts
from src.graph.guardrails.guardrail_helper import run_guardrail
from src.graph.state import State

logger = logging.getLogger(__name__)


def output_guardrail_node(state: State) -> dict:
    """
    Checks the agent's answer before it is returned to the user.
    """
    review_message = HumanMessage(content=(
        f"User message:\n{state['input_message']}\n\n"
        f"Assistant answer:\n{state['last_message']}"
    ))
    guardrail_output = run_guardrail(
        [SystemMessage(content=prompts.get_output_guardrail_prompt())],
        review_message, state["session_id"])
    logger.info("OUTPUT GUARDRAIL -> %s %s",
                guardrail_output.result, guardrail_output.reason)

    return {"output_guardrail_result": guardrail_output.model_dump()}
