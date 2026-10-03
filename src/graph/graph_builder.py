from langgraph.graph import END, START, StateGraph

# Import the nodes (agents, guardrails, subgraphs) and the routers of your flow
from src.graph.agents.fallback_agent import fallback_node
from src.graph.agents.general_agent import general_agent_node
from src.graph.agents.input_classifier_agent import input_classifier_node
from src.graph.finalize_turn import finalize_turn
from src.graph.guardrails.input_guardrail import input_guardrail_node
from src.graph.guardrails.output_guardrail import output_guardrail_node
from src.graph.init_agents import init_agents
from src.graph.routers.guardrail_router import (
    input_guardrail_router,
    output_guardrail_router,
)
from src.graph.routers.input_router import input_router
from src.graph.state import State
from src.graph.subgraphs.tool_subgraph.builder import subgraph as tool_subgraph

# Starts the graph builder
graph_builder = StateGraph(State)

"""
Under this section, you'll build your conversational flow here.
Here you'll add your nodes and the functions that they'll call, as
stated above. The graph is compiled in src/services/graph_service.py,
together with the Postgres checkpointer.
"""
# First, add your nodes and the function they'll call
graph_builder.add_node("init_agents", init_agents)
graph_builder.add_node("input_guardrail", input_guardrail_node)
graph_builder.add_node("input_classifier", input_classifier_node)
graph_builder.add_node("general_agent", general_agent_node)
graph_builder.add_node("tool_subgraph", tool_subgraph)
graph_builder.add_node("output_guardrail", output_guardrail_node)
graph_builder.add_node("fallback", fallback_node)
graph_builder.add_node("finalize_turn", finalize_turn)

# This is the start of the Graph. Here you'll make the paths your flow will take
graph_builder.add_edge(START, "init_agents")
graph_builder.add_edge("init_agents", "input_guardrail")
graph_builder.add_conditional_edges(
    "input_guardrail", input_guardrail_router,
    {
        "pass": "input_classifier",
        "fail": "fallback",
    }
)
graph_builder.add_conditional_edges(
    "input_classifier", input_router,
    {
        "general_agent": "general_agent",
        "tool_agent": "tool_subgraph",
    }
)
graph_builder.add_edge("general_agent", "output_guardrail")
graph_builder.add_edge("tool_subgraph", "output_guardrail")
graph_builder.add_conditional_edges(
    "output_guardrail", output_guardrail_router,
    {
        "pass": "finalize_turn",
        "fail": "fallback",
    }
)
graph_builder.add_edge("fallback", "finalize_turn")
graph_builder.add_edge("finalize_turn", END)
