from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GuardrailModel(BaseModel):
    """guardrail_output returned by the input and output guardrails."""

    result: Literal["PASS", "FAIL"] = Field(
        description="PASS if the content is allowed, FAIL if it breaks a rule.")
    reason: str = Field(
        default="", description="One short sentence explaining a FAIL. Empty on PASS.")

    @model_validator(mode="after")
    def check_reason(self):
        if self.result == "FAIL" and not self.reason:
            raise ValueError("Reason must be provided when result is FAIL.")
        if self.result == "PASS" and self.reason:
            self.reason = ""
        return self
