from src.graph import prompts
from src.graph.state import State
from src.utils.memory_helper import system_prompt


def init_agents(state: State) -> dict:
    """
    This function will be the first node of every turn. Resets values and refreshes each agent's
    system prompt.
    """
    return {
        "route": "",
        "last_message": "",
        "input_guardrail_result": {},
        "output_guardrail_result": {},
        "input_guardrail_agent": [
            system_prompt("input_guardrail_agent", prompts.get_input_guardrail_prompt())],
        "input_classifier": [
            system_prompt("input_classifier", prompts.get_input_classifier_prompt())],
        "general_agent": [
            system_prompt("general_agent", prompts.get_general_agent_prompt(state.get("shared_history")))],
    }
