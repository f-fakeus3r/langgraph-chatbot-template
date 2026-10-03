import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.graph.schemas import GuardrailModel
from src.services.chat_models import get_llm_guardrail_model
from src.utils.memory_helper import trim_messages_for_llm

logger = logging.getLogger(__name__)


def run_guardrail(history: list[BaseMessage], message: HumanMessage, session_id: str) -> GuardrailModel:
    """
    Ask the guardrail model for a structured PASS/FAIL output.
    Fails closed: if the call errors or returns no output, the result is FAIL,
    so unchecked content never goes through.
    """
    llm_guardrail_structured = get_llm_guardrail_model(
        session_id).with_structured_output(GuardrailModel, method="json_schema")
    try:
        guardrail_output = llm_guardrail_structured.invoke(
            trim_messages_for_llm([*history, message]))
    except Exception:
        logger.exception("Guardrail call failed")
        guardrail_output = None
    if isinstance(guardrail_output, GuardrailModel):
        return guardrail_output
    return GuardrailModel(result="FAIL", reason="The guardrail could not produce a guardrail_output.")


def guardrail_output_message(guardrail_output: GuardrailModel) -> AIMessage:
    """Output saved in the guardrail's own history, after the checked message."""
    content = guardrail_output.result if guardrail_output.result == "PASS" else f"FAIL: {guardrail_output.reason}"
    return AIMessage(content=content)
