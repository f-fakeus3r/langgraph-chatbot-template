from langchain_core.tools import tool

"""
Placeholder tools. Each one shows a different argument type.
Replace them with your own logic.
"""


@tool
def example_tool_one(query: str) -> str:
    """Example placeholder tool that takes a text query.
    Replace this docstring: the LLM reads it to decide when to call the tool."""
    return f"example_tool_one received: {query}"


@tool
def example_tool_two(item_id: int) -> str:
    """Example placeholder tool that takes a numeric ID.
    Replace this docstring: the LLM reads it to decide when to call the tool."""
    return f"example_tool_two received item_id: {item_id}"


@tool
def example_tool_three(name: str, limit: int = 5) -> str:
    """Example placeholder tool with an optional argument.
    Replace this docstring: the LLM reads it to decide when to call the tool."""
    return f"example_tool_three received name: {name}, limit: {limit}"


"""
Grouping related tools in a list lets the agent and the ToolNode register
all of them at once with *EXAMPLE_TOOLS.
"""
EXAMPLE_TOOLS = [example_tool_one, example_tool_two, example_tool_three]
