from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):

    # Request data set by GraphService.process_message
    input_message: str
    session_id: str
    username: str

    # Per-turn values, reset by init_agents at the start of every turn
    route: str  # agent that answers this turn: general_agent | tool_agent | fallback
    last_message: str  # answer sent to the user
    input_guardrail_result: dict
    output_guardrail_result: dict

    input_guardrail_agent: Annotated[list, add_messages]
    input_classifier: Annotated[list, add_messages]
    general_agent: Annotated[list, add_messages]
    tool_agent: Annotated[list, add_messages]
    # Every answered turn so each agent can see what was said to the others
    shared_history: Annotated[list, add_messages]
