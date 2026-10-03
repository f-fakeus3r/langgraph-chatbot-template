from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.graph.agents.tool_agent import tool_agent_node
from src.graph.state import State
from src.graph.subgraphs.tool_subgraph.prompt import update_prompt
from src.tools import ALL_TOOLS

"""
Example subgraph: a tool-calling agent packaged as its own graph. The parent
graph adds it as a single node ("tool_subgraph"). It shares the parent State,
and the tool loop works on the "tool_agent" message history.
"""

tool_node = ToolNode(tools=ALL_TOOLS, messages_key="tool_agent")


def tools_router(state: State) -> str:
    # "tools" if the agent's last message calls a tool, END otherwise
    return tools_condition(state, messages_key="tool_agent")


subgraph_builder = StateGraph(State)
subgraph_builder.add_node("update_prompt", update_prompt)
subgraph_builder.add_node("tool_agent", tool_agent_node)
subgraph_builder.add_node("tools", tool_node)

subgraph_builder.add_edge(START, "update_prompt")
subgraph_builder.add_edge("update_prompt", "tool_agent")
subgraph_builder.add_conditional_edges(
    "tool_agent",
    tools_router,
    {
        "tools": "tools",
        END: END,
    },
)
subgraph_builder.add_edge("tools", "tool_agent")

subgraph = subgraph_builder.compile()
