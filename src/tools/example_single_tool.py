from langchain_core.tools import tool

"""
Placeholder tool. Replace them with your own logic.
The function name becomes the tool name.
The docstring must be a detailed description so the LLM knows how and when
to use the tool.
"""


@tool
def example_tool_call(query: str) -> str:
    """Example placeholder tool.
    Replace this docstring: the LLM reads it to decide when to call the tool."""
    return f"example_tool_call received: {query}"
