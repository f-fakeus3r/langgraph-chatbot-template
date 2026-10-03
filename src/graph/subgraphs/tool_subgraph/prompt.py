from src.graph.prompts import get_language_and_behaviour_prompt
from src.graph.state import State
from src.utils.memory_helper import render_shared_history_context, system_prompt


def update_prompt(state: State) -> dict:
    """First node of the subgraph: refreshes the tool agent's system prompt."""
    content = get_tool_agent_prompt(state.get("shared_history"))
    return {"tool_agent": [system_prompt("tool_agent", content)]}


def get_tool_agent_prompt(shared_history=None) -> str:
    shared_history_context = render_shared_history_context(
        shared_history, "tool_agent")
    language_and_behavior = get_language_and_behaviour_prompt()
    # PLACEHOLDER: describe your documents and tools
    return f"""
    You are an assistant for <YOUR PRODUCT OR DOMAIN> with access to tools and to a knowledge base.

    Role:
    - You can call tools to take actions or fetch information.
    - Search the knowledge base before answering questions about <YOUR PRODUCT OR DOMAIN>.

    {shared_history_context}

    ## Tool Usage

    You have access to the following tools:

    ### qdrant_vector_search

    Use this tool for:
    - Finding information in the knowledge base (<DESCRIBE YOUR DOCUMENTS>)

    ### example_tool_call, example_tool_one, example_tool_two, example_tool_three

    Use these tools for:
    - Example usage of a tool (replace them with your own tools)

    ## Missing Information

    If the retrieved information is insufficient to answer the question:

    - Do not guess.
    - Do not fabricate information.
    - Try searching again with different words before giving up.
    - If you still can't find it, tell the user you don't have that information.

    {language_and_behavior}
    """
