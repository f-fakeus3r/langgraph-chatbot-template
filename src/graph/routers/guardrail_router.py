from src.graph.state import State


def input_guardrail_router(state: State) -> str:
    return _route(state["input_guardrail_result"])


def output_guardrail_router(state: State) -> str:
    return _route(state["output_guardrail_result"])


def _route(guardrail_result: dict) -> str:
    """Routes the conversation depending on guardrail's response."""
    if guardrail_result.get("result") == "PASS":
        return "pass"
    else:
        return "fail"
