import os
import re
import httpx

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import flash_client

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")

MAX_TOOL_ITERATIONS = 5

FLASH_SYSTEM = """You are a research worker. Your job is to investigate a specific angle of a problem.

Rules:
1. Briefly think step-by-step
2. When you need to verify something with code, output a Python code block:
   ```python
   # your code here
   ```
3. When you need web search, output a search query:
   ```search
   your search query here
   ```
4. After seeing tool results, provide a concise summary or output your final answer
5. Your final answer must start with "FINAL:" on its own line

CRITICAL: Be extremely concise. Do NOT repeat yourself. If you get stuck in a loop, stop and output your final results."""


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL)


def extract_search_queries(text: str) -> list[str]:
    return re.findall(r"```search\s*\n(.*?)\n```", text, re.DOTALL)


def has_final_answer(text: str) -> bool:
    return bool(re.search(r"^FINAL:\s*", text, re.MULTILINE))


async def run_code(code: str, worker_id: int) -> dict:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{SANDBOX_URL}/execute",
                json={"code": code.strip(), "timeout": 30},
            )
        result = resp.json()
        return {
            "worker_id": worker_id,
            "code": code.strip(),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", -1),
            "timed_out": result.get("timed_out", False),
        }
    except Exception as e:
        return {
            "worker_id": worker_id,
            "code": code.strip(),
            "stdout": "",
            "stderr": f"Tool execution error: {str(e)}",
            "exit_code": -2,
            "timed_out": False,
        }


async def run_search(query: str, worker_id: int) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query.strip(), "format": "json"},
            )
        data = resp.json()
        results = data.get("results", [])
        snippets = []
        for r in results[:10]:
            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")
            snippets.append(f"[{title}]({url})\n{snippet}")
        return "\n\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


async def flash_worker(state: DeepThinkState) -> dict:
    worker_id = state.get("worker_id", 0)
    prompt_data = state.get("prompt_data", {})
    prompt_text = prompt_data.get("text", "")
    prompt_type = prompt_data.get("type", "prove")

    writer = get_stream_writer()
    writer({"event": "flash_start", "worker": worker_id, "type": prompt_type})

    def on_token(chunk, is_reasoning=False):
        writer({"event": "token", "source": f"Worker {worker_id}", "text": chunk})

    conversation = [
        {"role": "system", "content": FLASH_SYSTEM},
    ]

    # If we have evaluation history, inject it as context to help the worker avoid previous mistakes
    if state.get("evaluation_history"):
        last_critique = state["evaluation_history"][-1]
        context_msg = (
            f"This is a RETRY attempt. Your previous attempt or your peers' attempts were critiqued by an advisor.\n"
            f"ADVISOR CRITIQUE:\n{last_critique}\n\n"
            f"Please take this critique into account, fix any logic or code errors, and provide a improved response."
        )
        conversation.append({"role": "user", "content": context_msg})

    conversation.append({"role": "user", "content": prompt_text})

    execution_logs = []
    final_response = ""
    worker_timed_out = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        resp = await flash_client.invoke(
            conversation,
            on_token=on_token,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )

        if resp.timed_out:
            writer(
                {"event": "flash_timeout", "worker": worker_id, "iteration": iteration}
            )
            final_response = (
                f"[TIMEOUT on iteration {iteration}] Partial: {resp.partial}"
            )
            worker_timed_out = True
            break

        final_response = resp.content
        conversation.append({"role": "assistant", "content": resp.content})

        if has_final_answer(resp.content):
            break

        codes = extract_code_blocks(resp.content)
        searches = extract_search_queries(resp.content)

        if not codes and not searches:
            break

        tool_results = []

        for code in codes:
            writer(
                {"event": "code_executing", "worker": worker_id, "iteration": iteration}
            )
            log = await run_code(code, worker_id)
            execution_logs.append(log)
            status = (
                "SUCCESS"
                if log["exit_code"] == 0
                else f"FAILED (exit {log['exit_code']})"
            )
            tool_results.append(
                f"Code execution {status}:\nstdout:\n{log['stdout']}\nstderr:\n{log['stderr']}"
            )

        for query in searches:
            writer({"event": "searching", "worker": worker_id, "query": query})
            search_result = await run_search(query, worker_id)
            execution_logs.append(
                {
                    "worker_id": worker_id,
                    "code": f"SEARCH: {query}",
                    "stdout": search_result[:2000],
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                }
            )
            tool_results.append(
                f"Search results for '{query}':\n{search_result[:1500]}"
            )

        if tool_results:
            conversation.append(
                {
                    "role": "user",
                    "content": "Tool results:\n"
                    + "\n---\n".join(tool_results)
                    + "\n\nContinue your reasoning or output your FINAL answer.",
                }
            )
        else:
            break

    output_text = final_response if has_final_answer(final_response) else final_response

    writer({"event": "flash_done", "worker": worker_id, "type": prompt_type})

    return {
        "flash_outputs": [
            {
                "worker_id": worker_id,
                "prompt_type": prompt_type,
                "response": output_text,
                "timed_out": worker_timed_out,
            }
        ],
        "execution_logs": execution_logs,
    }
