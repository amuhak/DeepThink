import json

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import pro_client


EVALUATOR_SYSTEM = """You are the Senior Advisor — a ruthless technical lead and research evaluator.

You will receive the original problem, current plan, worker outputs, execution logs, and evaluation history.

CRITICAL ASSESSMENT RULES:
1. PRIORITIZE EVIDENCE OVER VIBES: A technical report with specific URLs, paper titles (e.g., "ICLR 2026"), or GitHub handles is 10x more valuable than a "plausible" or "sophisticated" explanation. 
2. EPISTEMIC HUMILITY: You must accept that there are technologies, papers, and libraries released AFTER your training data. If workers provide empirical proof (links, code snippets, paper names), you MUST defer to that evidence even if it contradicts your internal "memory" or seems unlikely.
3. DETECT DOMAIN CONFLICTS: If workers disagree on the domain, do NOT guess. Identify which worker provided a specific, verifiable URL and investigate that source.
4. CONFLICT RESOLUTION: If you see a major domain contradiction, your PIVOT must be "Verify Primary Source." Ask the workers to find the official landing page, research paper, or author profile.
5. NO HALLUCINATION: Do NOT fill in technical details from your memory to "fix" worker outputs. If they haven't found it, pivot them to find it.

SOLVED STATE - TECHNICAL REPORT REQUIREMENTS:
- When you reach SOLVED, your "final_answer" MUST be an exhaustive, elite technical report.
- FORMAT: Use professional Markdown with clear sections.
- MATHEMATICS: Use LaTeX (e.g., $x^2$ or $$E=mc^2$$) for all formulas.
- CODE: Include any relevant algorithms or code snippets discovered.
- ARCHITECTURE: Describe the system/logic in high-fidelity detail.
- NO SUMMARIES: Never provide a high-level summary if technical details were discovered.

Output must be valid JSON with this exact structure:
{
  "status": "SOLVED" | "RETRY" | "PIVOT",
  "critique": "Your Senior Engineer feedback. Be blunt, tactical, and helpful.",
  "final_answer": "The exhaustive technical report (only if SOLVED). Use LaTeX and code blocks."
}

Status guidelines:
- SOLVED: Goal achieved with verified technical depth.
- RETRY: Small fixable errors.
- PIVOT: Strategy change needed. If workers are stuck, give them a new tactical lead.

Return raw JSON only."""


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


async def advisor_evaluator(state: DeepThinkState) -> dict:
    writer = get_stream_writer()
    writer({"event": "evaluating", "loop": state.get("loop_count", 0)})

    def on_token(chunk, is_reasoning=False):
        if not is_reasoning:
            writer({"event": "token", "source": "Evaluator", "text": chunk})

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
    ]

    # Build the evaluation prompt
    eval_parts = []

    if state.get("chat_history"):
        history = state["chat_history"]
        if history and history[-1]["content"] == state["user_prompt"]:
            history = history[:-1]

        if history:
            history_str = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history]
            )
            eval_parts.append(f"## Previous Conversation Context\n{history_str}")

    eval_parts.append(f"## Original Problem\n{state['user_prompt']}")
    eval_parts.append(f"## Current Plan\n{state.get('current_plan', 'No plan')}")

    if state.get("flash_outputs"):
        eval_parts.append("## Worker Outputs")
        for i, out in enumerate(state["flash_outputs"]):
            resp = out.get("response", "No output")
            import re

            final_match = re.search(r"^FINAL:\s*(.*)", resp, re.MULTILINE)
            if final_match:
                summary = f"FINAL: {final_match.group(1).strip()[:2000]}"
            else:
                summary = resp[-2000:] if len(resp) > 2000 else resp
            worker_id = out.get("worker_id", i)
            worker_type = out.get("prompt_type", "?")
            eval_parts.append(
                f"\n### Worker {worker_id} ({worker_type})\n"
                f"{'[TIMED OUT]' if out.get('timed_out') else summary}"
            )

    if state.get("execution_logs"):
        eval_parts.append("## Execution Logs")
        for i, log in enumerate(state["execution_logs"][-3:]):
            worker_id = log.get("worker_id", "?")
            eval_parts.append(
                f"Log {i} (W{worker_id}): exit={log.get('exit_code', '?')} stdout={log.get('stdout', '')[:100]}"
            )

    if state.get("evaluation_history"):
        eval_parts.append(
            f"## Previous Evaluation History\n"
            + "\n---\n".join(state["evaluation_history"][-3:])
        )

    messages.append(
        {
            "role": "user",
            "content": "\n\n".join(eval_parts),
        }
    )

    resp = await pro_client.invoke_json(
        messages,
        temperature=0.6,
        on_token=on_token,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    if resp.timed_out:
        writer(
            {"event": "decision", "status": "RETRY", "reason": "Evaluator timed out"}
        )
        return {
            "status": "RETRY",
            "evaluation_history": ["Evaluator timed out — retrying workers"],
        }

    try:
        data = parse_json_response(resp.content)
        status = data.get("status", "RETRY")
        critique = data.get("critique", "No critique provided")
        final_answer = data.get("final_answer", None)
    except (json.JSONDecodeError, KeyError) as e:
        status = "RETRY"
        critique = f"Evaluator JSON parse failed: {str(e)}. Raw: [{resp.content[:200]}]"
        final_answer = None

    writer({"event": "decision", "status": status, "reason": critique[:200]})
    print(f"[Evaluator] Status: {status}, Critique: {critique[:200]}", flush=True)
    print(f"[Evaluator] Raw response length: {len(resp.content)}", flush=True)

    result = {
        "status": status,
        "evaluation_history": [critique],
        "loop_count": state.get("loop_count", 0) + 1,
    }

    if status == "SOLVED" and final_answer is not None:
        result["final_answer"] = final_answer

    return result
