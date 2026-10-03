from langchain_core.messages import AIMessage

from src.graph.prompts import INPUT_BLOCKED_MESSAGE, OUTPUT_BLOCKED_MESSAGE
from src.graph.state import State


def fallback_node(state: State) -> dict:
    """
    Replaces the answer when a guardrail fails.

    - Input guardrail FAIL: no agent ran, reply with a refusal.
    - Output guardrail FAIL: an agent already answered.
    """
    if state["output_guardrail_result"].get("result") == "FAIL":
        agent = state["route"]
        blocked_answer = state[agent][-1]
        return {
            agent: [AIMessage(content=OUTPUT_BLOCKED_MESSAGE, id=blocked_answer.id)],
            "last_message": OUTPUT_BLOCKED_MESSAGE,
        }

    return {
        "route": "fallback",
        "last_message": INPUT_BLOCKED_MESSAGE,
    }
