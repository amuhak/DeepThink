import sys

sys.path.insert(0, "/app")
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TestState(TypedDict):
    value: str


async def node_a(state: TestState) -> dict:
    print("Node A running...", flush=True)
    return {"value": "from A"}


async def node_b(state: TestState) -> dict:
    print("Node B running...", flush=True)
    return {"value": state["value"] + " and B"}


async def test():
    print("Building simple graph...", flush=True)
    builder = StateGraph(TestState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()
    print("Graph built. Invoking...", flush=True)
    try:
        result = await asyncio.wait_for(graph.ainvoke({"value": ""}), timeout=5.0)
        print("Success! Result:", result, flush=True)
    except asyncio.TimeoutError:
        print("Timed out after 5 seconds!", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)


asyncio.run(test())
