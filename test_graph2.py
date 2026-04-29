import sys

sys.path.insert(0, "/app")
import asyncio
from graph import build_graph
from state import DeepThinkState


async def test():
    print("Building graph...", flush=True)
    g = build_graph()
    print("Graph built. Invoking...", flush=True)
    try:
        result = await asyncio.wait_for(
            g.ainvoke(
                {
                    "user_prompt": "What is 2+2?",
                    "chat_history": [{"role": "user", "content": "What is 2+2?"}],
                    "status": "RUNNING",
                    "loop_count": 0,
                    "flash_outputs": [],
                    "execution_logs": [],
                    "evaluation_history": [],
                    "flash_prompts": [],
                    "current_plan": "",
                    "final_answer": "",
                }
            ),
            timeout=5.0,
        )
        print("Success! Result:", str(result)[:200], flush=True)
    except asyncio.TimeoutError:
        print("Timed out after 5 seconds!", flush=True)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()


asyncio.run(test())
