import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import DeepThinkState
from nodes.advisor_planner import advisor_planner
from nodes.flash_worker import flash_worker
from nodes.advisor_evaluator import advisor_evaluator
from nodes.advisor_synthesizer import advisor_synthesizer


def route_to_workers(state: DeepThinkState) -> list[Send]:
    prompts = state.get("flash_prompts", [])
    print(f"[Graph] route_to_workers called with {len(prompts)} prompts, loop={state.get('loop_count', 0)}", flush=True)
    if not prompts:
        print(f"[Graph] WARNING: flash_prompts is empty!", flush=True)
    return [
        Send("flash_worker", {"worker_id": i, "prompt_data": p})
        for i, p in enumerate(prompts)
    ]


def route_after_eval(state: DeepThinkState):
    max_loops = int(os.environ.get("MAX_LOOPS", "10"))
    status = state.get("status", "RETRY")
    loop = state.get("loop_count", 1)
    
    if status == "SOLVED" or loop >= max_loops:
        return "advisor_synthesizer"
    elif status == "PIVOT":
        return "advisor_planner"
    else:
        return route_to_workers(state)


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
