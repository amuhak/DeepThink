import asyncio
import os
import sys

# Add orchestrator to path
sys.path.append(os.path.join(os.getcwd(), "orchestrator"))

from orchestrator.nodes.flash_worker import flash_worker
from orchestrator.state import DeepThinkState


async def main():
    state: DeepThinkState = {
        "user_prompt": "Scrape https://example.com and tell me what it says.",
        "prompt_data": {
            "text": "Scrape https://example.com and tell me what it says. Use the ```scrape``` tool.",
            "type": "prove",
        },
        "worker_id": 0,
        "loop_count": 0,
        "evaluation_history": [],
    }

    # We need to mock get_stream_writer
    import langgraph.config

    def mock_writer(event):
        # print(f"EVENT: {event.get('event')} - {event.get('url', event.get('query', ''))}")
        if event.get("event") == "token":
            print(event.get("text"), end="", flush=True)

    langgraph.config.get_stream_writer = lambda: mock_writer

    print("Running flash_worker...")
    result = await flash_worker(state)
    print("\n\nWorker Result:")
    print(result["flash_outputs"][0]["response"])


if __name__ == "__main__":
    asyncio.run(main())
