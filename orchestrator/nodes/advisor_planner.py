import os
import json

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import pro_client


PRO_PLANNER_SYSTEM = """You are the Advisor — a strategic research planner.

Your job is to take a complex problem and design a research plan using "Balanced Prompting".
You must generate exactly NUM_PROMPTS sub-prompts for parallel worker models:
- Half should attempt to PROVE the premise
- Half should attempt to REFUTE or find counterexamples

TOOL USAGE INSTRUCTIONS:
- Tell workers specifically to use ```search\\nquery\\n```, ```scrape\\nurl\\n```, or ```python\\ncode\\n``` blocks.
- **IMPORTANT:** Explicitly forbid workers from writing Python code to scrape (requests/BS4). Tell them to use the ```scrape``` tool instead as it bypasses bot detection.
- Remind them that ```scrape``` is the most powerful way to get full technical text, math, and code from a URL.
- They MUST NOT simulate results.

Output must be valid JSON with this exact structure:
{
  "plan": "High-level research strategy",
  "prompts": [
    {"text": "Investigate X by using the ```search``` tool for queries A and B. Then verify with ```python```...", "type": "prove"},
    {"text": "Direct instructions for worker...", "type": "refute"}
  ]
}

CRITICAL: Be extremely concise. Return raw JSON only."""



def parse_json_response(content: str) -> dict:
    content = content.strip()
    # Remove markdown code fences
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    # Find first { and matching }
    brace_start = content.find("{")
    if brace_start < 0:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    # Track nested braces to find matching closing brace
    depth = 0
    for i in range(brace_start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                return json.loads(content[brace_start : brace_end + 1])
    raise json.JSONDecodeError("No matching closing brace", content, brace_start)


async def advisor_planner(state: DeepThinkState) -> dict:
    writer = get_stream_writer()
    writer({"event": "planning", "loop": state.get("loop_count", 0) + 1})

    def on_token(chunk, is_reasoning=False):
        if not is_reasoning:
            writer({"event": "token", "source": "Planner", "text": chunk})

    n_explorers = int(os.environ.get("NUM_FLASH_EXPLORERS", "4"))
    system_prompt = PRO_PLANNER_SYSTEM.replace("NUM_PROMPTS", str(n_explorers))

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if state.get("evaluation_history"):
        critique = "\n---\n".join(state["evaluation_history"])
        messages.append(
            {
                "role": "user",
                "content": f"Previous attempt failed. Here is the evaluation history:\n{critique}\n\nPlease devise a NEW strategy and generate fresh prompts.",
            }
        )

    messages.append({"role": "user", "content": f"Problem: {state['user_prompt']}"})

    resp = await pro_client.invoke_json(
        messages,
        temperature=1.0,
        on_token=on_token,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
    )

    if resp.timed_out:
        n_prove = n_explorers // 2
        n_refute = n_explorers - n_prove
        fallback_prompts = [
            {
                "text": f"Prove the following by writing and executing Python code: {state['user_prompt']}",
                "type": "prove",
            }
            for _ in range(n_prove)
        ] + [
            {
                "text": f"Try to refute or find a counterexample for: {state['user_prompt']}",
                "type": "refute",
            }
            for _ in range(n_refute)
        ]
        return {
            "current_plan": "Fallback plan (Pro model timed out)",
            "flash_prompts": fallback_prompts,
            "status": "RUNNING",
        }

    try:
        data = parse_json_response(resp.content)
        prompts = data.get("prompts", [])
        plan = data.get("plan", "No plan provided")
    except (json.JSONDecodeError, KeyError):
        prompts = [
            {"text": f"Analyze and prove: {state['user_prompt']}", "type": "prove"},
            {"text": f"Try to refute: {state['user_prompt']}", "type": "refute"},
        ]
        plan = "Fallback plan (JSON parse failed)"

    writer({"event": "plan_generated", "plan": plan, "num_prompts": len(prompts)})

    return {
        "current_plan": plan,
        "flash_prompts": prompts,
        "status": "RUNNING",
    }
