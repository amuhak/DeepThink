import os
import re
import httpx

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import flash_client

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")
FIRECRAWL_URL = os.environ.get("FIRECRAWL_URL", "http://firecrawl-api:3002")

MAX_TOOL_ITERATIONS = 5

FLASH_SYSTEM = """You are a high-performance research worker. You have direct access to a Python sandbox, a Web Search engine, and a URL Scraper.

TOOL RULES:
1. PYTHON: To execute code, use:
   ```python
   # your code here
   ```
   **RESTRICTION:** Do NOT write code to scrape websites (e.g., using requests, beautifulsoup, selenium). Use the SCRAPE tool instead. Python scraping is often blocked; the SCRAPE tool uses a professional headless browser.

2. SEARCH: To search the web, use:
   ```search
   your search query here
   ```
3. SCRAPE: To read the full content of a specific URL (arXiv, GitHub, documentation), use:
   ```scrape
   https://example.com/target-page
   ```
   **CRITICAL:** Use SCRAPE whenever you find a promising technical link to get full math, code, and details.

4. EXECUTION IS REAL: Do NOT simulate or "thought-block" your tools.
5. NO LOOPING: If you have tried searching/scraping twice with no results, admit it and output your FINAL summary.

OUTPUT FORMAT:
- Briefly think step-by-step.
- Use tools as needed.
- Your final answer must start with "FINAL:" on its own line.
- **SOURCE ATTRIBUTION:** For every key fact, you MUST state where you found it (e.g., "Source: [GitHub URL]" or "Source: [Paper Title]").
- Always try to extract mathematical formulas and code snippets.
"""


def extract_code_blocks(text: str) -> list[str]:
    # Matches ```python ... ``` with optional language tag and flexible whitespace
    return re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)


def extract_search_queries(text: str) -> list[str]:
    return re.findall(r"```search\s*(.*?)\s*```", text, re.DOTALL)


def extract_scrape_urls(text: str) -> list[str]:
    return re.findall(r"```scrape\s*(.*?)\s*```", text, re.DOTALL)


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
                params={"q": query.strip(), "format": "json", "pageno": 1},
            )
        data = resp.json()
        results = data.get("results", [])
        # Log search results for debugging
        print(
            f"[Worker {worker_id}] Search '{query}': {len(results)} results from {data.get('engines', [])}"
        )
        snippets = []
        for r in results:  # Use ALL results, not just first 10
            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")
            engine = r.get("engine", "unknown")
            if title and url:
                snippets.append(f"[{title}]({url})\nSource: {engine}\n{snippet}")
        if not snippets:
            return "No results found."
        return f"Found {len(snippets)} results:\n\n" + "\n\n---\n\n".join(snippets)
    except Exception as e:
        return f"Search error: {str(e)}"


async def run_scrape(url: str, worker_id: int) -> str:
    url = url.strip()
    # Try local Firecrawl first
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{FIRECRAWL_URL}/v1/scrape",
                json={"url": url, "formats": ["markdown"]},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("data", {}).get("markdown", "")[:15000]
    except Exception:
        pass

    # Fallback to r.jina.ai (User suggestion)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(jina_url)
        return resp.text[:10000]
    except Exception as e:
        return f"Scrape error for {url}: {str(e)}"


async def flash_worker(state: DeepThinkState) -> dict:
    worker_id = state.get("worker_id", 0)
    prompt_data = state.get("prompt_data", {})
    prompt_text = prompt_data.get("text", "")
    prompt_type = prompt_data.get("type", "prove")

    writer = get_stream_writer()
    writer({"event": "flash_start", "worker": worker_id, "type": prompt_type})

    conversation = [
        {"role": "system", "content": FLASH_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    execution_logs = []
    used_queries = set()
    final_response = ""
    worker_timed_out = False

    def on_token(chunk, is_reasoning=False):
        writer({"event": "token", "source": f"Worker {worker_id}", "text": chunk})

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
        scrapes = extract_scrape_urls(resp.content)

        if not codes and not searches and not scrapes:
            break

        tool_results = []

        # Stop if we are repeating searches/scrapes
        repetitive = False
        for q in searches + scrapes:
            if q in used_queries:
                repetitive = True
                break
            used_queries.add(q)

        if repetitive:
            tool_results.append(
                "ERROR: Repetitive tool call detected. You are repeating a query that already failed or yielded no new info. Please STOP and provide your FINAL summary based on what you already found."
            )

        if not repetitive:
            for code in codes:
                writer(
                    {
                        "event": "code_executing",
                        "worker": worker_id,
                        "iteration": iteration,
                    }
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
                        "stdout": search_result[:10000],
                        "stderr": "",
                        "exit_code": 0,
                        "timed_out": False,
                    }
                )
                tool_results.append(
                    f"Search results for '{query}':\n{search_result[:5000]}"
                )

            for url in scrapes:
                writer({"event": "scraping", "worker": worker_id, "url": url})
                scrape_result = await run_scrape(url, worker_id)
                execution_logs.append(
                    {
                        "worker_id": worker_id,
                        "code": f"SCRAPE: {url}",
                        "stdout": scrape_result[:2000],
                        "stderr": "",
                        "exit_code": 0,
                        "timed_out": False,
                    }
                )
                tool_results.append(f"Scraped content from {url}:\n{scrape_result}")

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
