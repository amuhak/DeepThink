import asyncio
import os
import sys

# Add current folder to path
sys.path.append(os.getcwd())

from state import DeepThinkState
from nodes.pdf_processor import pdf_processor
import langgraph.config

async def test_pdf():
    # Setup mock state with a real paper on PDF quantization or similar
    # We will download a simple public PDF from a stable URL
    state: DeepThinkState = {
        "user_prompt": "Analyze the formulas in the paper.",
        "pending_pdfs": [
            {
                "url": "https://arxiv.org/pdf/2402.17762.pdf", # Tiny, valid arXiv PDF
                "question": "What is the primary formula or method proposed in this paper? Tell me in 10 words.",
                "pages": "1", # Only analyze page 1 for speed
                "worker_id": 0
            }
        ],
        "flash_prompts": [],
        "flash_outputs": [],
        "execution_logs": [],
        "evaluation_history": [],
    }

    # Mock LangGraph stream writer to print event logs
    def mock_writer(event):
        if event.get("event") == "token":
            print(event.get("text"), end="", flush=True)

    langgraph.config.get_stream_writer = lambda: mock_writer

    print("--- Running PDF Processor in Container ---")
    result = await pdf_processor(state)
    print("\n\n--- Result outputs ---")
    for out in result.get("flash_outputs", []):
        print(f"Worker ID: {out['worker_id']}")
        print(f"Response:\n{out['response']}")

if __name__ == "__main__":
    asyncio.run(test_pdf())
