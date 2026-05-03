import os
import json

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import pro_client


PRO_PLANNER_SYSTEM = """You are the Advisor — a strategic research planner.

Your job is to design a research plan using "Two-Phase Discovery" to ensure accuracy and prevent domain-specific hallucinations.

PHASE 1: NEUTRAL SCOUTING
- Never assume the category, industry, or domain of a subject based on its name alone.
- Dedicate at least one worker to perform broad, unconstrained searches to identify the "ground truth" context (e.g., "What is [Subject]?", "[Subject] technical overview", "[Subject] official documentation").
- This phase prevents the "Confirmation Bias Trap" where the system searches for evidence of a guess rather than the reality.

PHASE 2: BALANCED DEEP-DIVE
- Once the context is established, generate NUM_PROMPTS sub-prompts for parallel workers:
- PROVE: Half should extract core logic, implementation details, formulas, and primary evidence.
- REFUTE: Half should look for edge cases, limitations, competing technologies, or verify if the subject is a misnomer/hallucination.

TOOL USAGE INSTRUCTIONS:
- Instruct workers to use ```search\\nquery\\n```, ```scrape\\nurl\\n```, or ```python\\ncode\\n``` blocks.
- **SCRAPE IS MANDATORY:** Remind workers that ```scrape``` is the only way to get full text, math, and code from a URL.
- **NO PYTHON SCRAPING:** Explicitly forbid writing requests/BS4 code; use the ```scrape``` tool.

Output must be valid JSON:
{
  "plan": "High-level strategy (Scout for context -> Parallel deep-dive)",
  "prompts": [
    {"text": "Neutral Scout: Search for [Subject] to identify its domain and primary sources...", "type": "prove"},
    {"text": "Parallel Worker: Investigate [Specific Aspect] using...", "type": "refute"}
  ]
}

CRITICAL: Stay neutral. Avoid premature commitment to a domain. Return raw JSON only."""


def parse_json_response(content: str) -> dict:
    content = content.strip()
    # Remove markdown code fences
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
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
                json_string = content[brace_start : brace_end + 1]
                import re
                # Fix LaTeX: escape single backslashes before common commands
                json_string = re.sub(r'(?<!\\)\\([a-zA-Z]+)', r'\\\\\1', json_string)
                # Fix remaining invalid escapes
                json_string = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_string)
                return json.loads(json_string, strict=False)
    # Try to fix truncated JSON by adding missing braces
    truncated = content[brace_start:]
    while depth > 0:
        truncated += "}"
        depth -= 1
    # Try parsing the fixed JSON
    try:
        return json.loads(truncated)
    except json.JSONDecodeError:
        pass
    raise json.JSONDecodeError("No matching closing brace", content, brace_start)


async def advisor_planner(state: DeepThinkState) -> dict:
    writer = get_stream_writer()
    writer({"event": "planning"})

    print(f"[Planner] Starting...", flush=True)
    with open("/tmp/planner.log", "a") as f:
        f.write("[Planner] Starting...\n")

    n_explorers = int(os.environ.get("NUM_FLASH_EXPLORERS", "4"))
    n_prove = n_explorers // 2
    n_refute = n_explorers - n_prove

    # Build the messages for the LLM
    messages = [
        {"role": "system", "content": PRO_PLANNER_SYSTEM},
    ]

    if state.get("chat_history"):
        # Exclude the last message if it matches the current user_prompt to avoid duplication
        history = state["chat_history"]
        if history and history[-1]["content"] == state["user_prompt"]:
            history = history[:-1]

        # Compact context: only keep the last 4 messages of history
        history = history[-4:]

        if history:
            history_str = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history]
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"RESEARCH CONTEXT (Previous turns):\n{history_str}",
                }
            )

    if state.get("evaluation_history"):
        critique = "\n---\n".join(state["evaluation_history"])
        messages.append(
            {
                "role": "user",
                "content": f"Previous attempt failed. Here is the evaluation history:\n{critique}\n\nPlease devise a NEW strategy and generate fresh prompts.",
            }
        )

    messages.append({"role": "user", "content": f"Problem: {state['user_prompt']}"})

    print(f"[Planner] Calling LLM...", flush=True)
    with open("/tmp/planner.log", "a") as f:
        f.write("[Planner] Calling LLM...\n")

    def on_token(chunk, is_reasoning=False):
        writer({"event": "token", "source": "Planner", "text": chunk})

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

    # TEMP: Skip LLM call for testing
    # import json
    # fake_resp_content = json.dumps({"plan": "Test plan", "prompts": [{"text": "Test prompt", "type": "prove"}]})

    # Create a fake response
    # class FakeResp:
    #     def __init__(self):
    #         self.content = fake_resp_content
    #         self.timed_out = False
    #
    # resp = FakeResp()

    print(f"[Planner] LLM response received. Timed out: {resp.timed_out}", flush=True)
    with open("/tmp/planner.log", "a") as f:
        f.write(f"[Planner] LLM response received. Timed out: {resp.timed_out}\n")

    # Parse output
    raw = resp.content.strip()
    print(f"[Planner] Raw response length: {len(raw)}", flush=True)
    with open("/tmp/planner.log", "a") as f:
        f.write(f"[Planner] Raw response length: {len(raw)}\n")

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
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Planner] JSON parse failed: {e}. Raw: {raw[:500]}", flush=True)
        prompts = [
            {"text": f"Analyze and prove: {state['user_prompt']}", "type": "prove"},
            {"text": f"Try to refute: {state['user_prompt']}", "type": "refute"},
        ]
        plan = "Fallback plan (JSON parse failed)"

    writer({"event": "plan_generated", "plan": plan, "num_prompts": len(prompts)})

    print(
        f"[Planner] Returning {len(prompts)} prompts. Plan: {plan[:50]}...", flush=True
    )
    with open("/tmp/planner.log", "a") as f:
        f.write(f"[Planner] Returning {len(prompts)} prompts. Plan: {plan[:50]}...\n")
    return {
        "current_plan": plan,
        "flash_prompts": prompts,
        "status": "RUNNING",
    }
