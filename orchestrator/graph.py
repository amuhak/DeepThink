import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import DeepThinkState
from nodes.advisor_planner import advisor_planner
from nodes.flash_worker import flash_worker
from nodes.advisor_evaluator import advisor_evaluator
from nodes.advisor_synthesizer import advisor_synthesizer


def route_to_workers(state: DeepThinkState) -> list[Send]:
    return [
        Send("flash_worker", {"worker_id": i, "prompt_data": p})
        for i, p in enumerate(state["flash_prompts"])
    ]


def route_after_eval(state: DeepThinkState):
    max_loops = int(os.environ.get("MAX_LOOPS", "10"))

    if state["status"] == "SOLVED":
        return "advisor_synthesizer"
    elif state["loop_count"] >= max_loops:
        return END
    elif state["status"] == "PIVOT":
        return "advisor_planner"
    else:
        return [
            Send("flash_worker", {"worker_id": i, "prompt_data": p})
            for i, p in enumerate(state["flash_prompts"])
        ]


def build_graph() -> StateGraph:
    builder = StateGraph(DeepThinkState)

    builder.add_node("advisor_planner", advisor_planner)
    builder.add_node("flash_worker", flash_worker)
    builder.add_node("advisor_evaluator", advisor_evaluator)
    builder.add_node("advisor_synthesizer", advisor_synthesizer)

    builder.add_edge(START, "advisor_planner")
    builder.add_conditional_edges("advisor_planner", route_to_workers)
    builder.add_edge("flash_worker", "advisor_evaluator")
    builder.add_conditional_edges("advisor_evaluator", route_after_eval)
    builder.add_edge("advisor_synthesizer", END)

    return builder.compile()
