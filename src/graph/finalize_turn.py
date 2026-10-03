from src.graph.state import State
from src.utils.memory_helper import append_shared_turn


def finalize_turn(state: State) -> dict:
    """
    Last node of every turn: saves the turn in the shared history, so the
    other agents see it in their system prompt from the next turn on.
    """
    if state["route"] == "fallback":
        # Blocked by the input guardrail: keep that message out of every prompt
        return {}

    return {
        "shared_history": append_shared_turn(
            state["route"], state["input_message"], state["last_message"]),
    }
