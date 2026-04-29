import asyncio
import sys

sys.path.insert(0, "orchestrator")

from llm_client import pro_client

PRO_PLANNER_SYSTEM = """You are the Advisor — a strategic research planner.

Your job is to take a complex problem and design a research plan using "Balanced Prompting".
You must generate exactly 4 sub-prompts for parallel worker models:
- Half should attempt to PROVE the premise
- Half should attempt to REFUTE or find counterexamples

Each prompt should instruct the worker to:
1. Think through the problem step by step
2. Write Python code to verify claims (use sympy for math, numerical methods for computation)
3. Use web search if background knowledge is needed
4. Return a final verdict with reasoning

Output must be valid JSON with this exact structure:
{
  "plan": "High-level research strategy",
  "prompts": [
    {"text": "Prompt for worker...", "type": "prove"},
    {"text": "Prompt for worker...", "type": "refute"}
  ]
}

Do NOT include markdown code fences around the JSON. Return raw JSON only."""

messages = [
    {"role": "system", "content": PRO_PLANNER_SYSTEM},
    {"role": "user", "content": "Problem: What is 2+2?"},
]


async def test():
    resp = await pro_client.invoke_json(messages, temperature=0.3)
    print(f"Timed out: {resp.timed_out}")
    print(f"Partial: {resp.partial}")
    print(f"Content: [{resp.content}]")
    print(f"Content length: {len(resp.content)}")


asyncio.run(test())
