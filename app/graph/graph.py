from typing import Any, cast
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from app.graph.state import GraphState
from app.graph.nodes import (
    research_node,
    director_node,
    shot_loop_node,
    assembly_node,
)

def build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    builder = StateGraph(GraphState)

    # ── Register nodes ──────────────────────────────────────────────────
    builder.add_node("research",  research_node)
    builder.add_node("director",  director_node)
    builder.add_node("shot_loop", shot_loop_node)
    builder.add_node("assembly",  assembly_node)

    # ── Define edges (linear pipeline) ─────────────────────────────────
    builder.set_entry_point("research")
    builder.add_edge("research",  "director")
    builder.add_edge("director",  "shot_loop")
    builder.add_edge("shot_loop", "assembly")
    builder.add_edge("assembly",  END)

    return cast(CompiledStateGraph[Any, Any, Any, Any], builder.compile())


# Singleton — import this in your API layer
graph = build_graph()
