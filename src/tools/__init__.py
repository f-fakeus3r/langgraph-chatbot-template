from src.tools.example_multiple_tools import EXAMPLE_TOOLS
from src.tools.example_single_tool import example_tool_call
from src.tools.qdrant_rag_tool import qdrant_vector_search

"""
Register your tools here. ALL_TOOLS is bound to the tool agent and executed
by the ToolNode of the tool subgraph, so both always see the same list.
"""
ALL_TOOLS = [qdrant_vector_search, *EXAMPLE_TOOLS, example_tool_call]
