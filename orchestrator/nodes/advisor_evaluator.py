import json

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import pro_client


EVALUATOR_SYSTEM = """You are the Senior Advisor — a ruthless technical lead and research evaluator.

You will receive the original problem, current plan, worker outputs, execution logs, and evaluation history.

CRITICAL ASSESSMENT RULES:
1. DETECT LOOPS: If workers are repeating failed searches or scripts, you MUST intervene.
2. SENIOR COACHING: In your "critique", do NOT just say "try again". Act like a Senior Engineer:
   - If a worker fails to scrape a page, suggest they search for a GitHub repo, a PDF link, or try a different search engine query.
   - If code fails, suggest a specific library (e.g., "try using PyPDF2" or "check the shape of the matrix").
3. NO HALLUCINATION: If the workers haven't found the specific math/code, do NOT fill it in from your own memory unless you are 100% sure. Instead, PIVOT the workers to find it.
4. SCRAPING EFFICIENCY: If workers are writing Python code (requests/BS4) to scrape, you MUST critique them and tell them to use the ```scrape``` tool. The sandbox does not have scraping libraries; the ```scrape``` tool is the only supported way to read URLs.

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

    eval_parts.append(f"## Original Problem\n{state['user_prompt']}")
    eval_parts.append(f"## Current Plan\n{state.get('current_plan', 'No plan')}")

    if state.get("flash_outputs"):
        eval_parts.append("## Worker Outputs")
        for i, out in enumerate(state["flash_outputs"]):
            resp = out.get("response", "No output")
            import re

            final_match = re.search(r"^FINAL:\s*(.*)", resp, re.MULTILINE)
            if final_match:
                summary = f"FINAL: {final_match.group(1).strip()[:500]}"
            else:
                # Just take last 300 chars as summary
                summary = resp[-300:] if len(resp) > 300 else resp
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

    writer({"event": "evaluation_complete"})

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

    result = {
        "status": status,
        "evaluation_history": [critique],
        "loop_count": state.get("loop_count", 0) + 1,
    }

    if status == "SOLVED" and final_answer is not None:
        result["final_answer"] = final_answer

    return result
