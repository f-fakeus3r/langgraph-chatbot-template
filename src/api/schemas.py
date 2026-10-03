from pydantic import BaseModel, Field


class UserData(BaseModel):
    input_message: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    username: str


class MessageResponse(BaseModel):
    input_message: str
    output_message: str
    route: str  # agent that answered: general_agent | tool_agent | fallback
    input_guardrail: str  # PASS | FAIL
    output_guardrail: str  # PASS | FAIL, empty when the input guardrail failed
