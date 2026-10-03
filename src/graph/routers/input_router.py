from src.graph.state import State


def input_router(state: State) -> str:
    """Sends the turn to the agent picked by the input classifier."""
    return state["route"]
