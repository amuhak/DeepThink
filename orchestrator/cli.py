import argparse
import asyncio
import sys

from graph import build_graph


async def run_cli(
    prompt: str, verbose: bool = False, max_loops: int = 10, num_explorers: int = 4
):
    import os

    os.environ["MAX_LOOPS"] = str(max_loops)
    os.environ["NUM_FLASH_EXPLORERS"] = str(num_explorers)

    graph = build_graph()

    print(f"\n{'='*60}")
    print(f"DeepThink CLI")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Explorers: {num_explorers} | Max loops: {max_loops}")
    print(f"{'='*60}\n")

    initial_state = {
        "user_prompt": prompt,
        "status": "RUNNING",
        "loop_count": 0,
        "flash_outputs": [],
        "execution_logs": [],
        "evaluation_history": [],
        "flash_prompts": [],
        "current_plan": "",
    }

    try:
        async for part in graph.astream(
            initial_state,
            stream_mode=["updates", "custom"],
            version="v2",
            config={"recursion_limit": 100},
        ):
            ptype = part.get("type", "")
            data = part.get("data", {})

            if ptype == "custom":
                event = data.get("event", "")
                if verbose:
                    print(
                        f"[EVENT] {event}: {json.dumps({k: v for k, v in data.items() if k != 'event'})}"
                    )
                elif event == "planning":
                    print(f"  Loop {data.get('loop', '?')} — Planning...")
                elif event == "plan_generated":
                    print(f"  Plan: {data.get('plan', '')[:200]}")
                elif event == "flash_start":
                    print(
                        f"  Worker {data.get('worker', '?')} ({data.get('type', '?')}) started"
                    )
                elif event == "flash_done":
                    print(
                        f"  Worker {data.get('worker', '?')} ({data.get('type', '?')}) done"
                    )
                elif event == "flash_timeout":
                    print(f"  Worker {data.get('worker', '?')} timed out")
                elif event == "code_executing":
                    print(f"  Worker {data.get('worker', '?')} → executing code")
                elif event == "searching":
                    print(
                        f"  Worker {data.get('worker', '?')} → searching: {data.get('query', '')[:60]}"
                    )
                elif event == "evaluating":
                    print(f"  Evaluating results...")
                elif event == "decision":
                    print(
                        f"  Decision: {data.get('status', '?')} — {data.get('reason', '')[:150]}"
                    )

            elif ptype == "updates":
                for node_name, update in data.items():
                    if isinstance(update, dict):
                        if (
                            node_name == "advisor_evaluator"
                            and update.get("status") == "SOLVED"
                        ):
                            answer = update.get("final_answer", "No answer")
                            print(f"\n{'='*60}")
                            print(f"ANSWER:")
                            print(f"{'='*60}")
                            print(answer)
                            print(f"{'='*60}\n")
                            return
                        if verbose:
                            print(
                                f"[UPDATE] {node_name}: {json.dumps(update, default=str)[:300]}"
                            )

        print("\n[Graph completed — no SOLVED status reached]")
        if verbose:
            print("[Run with --verbose for full state dump]")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="DeepThink CLI")
    parser.add_argument("prompt", help="Research prompt")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument("--loops", type=int, default=10, help="Max iteration loops")
    parser.add_argument(
        "--explorers", type=int, default=4, help="Number of parallel Flash workers"
    )
    args = parser.parse_args()

    asyncio.run(run_cli(args.prompt, args.verbose, args.loops, args.explorers))


if __name__ == "__main__":
    main()
