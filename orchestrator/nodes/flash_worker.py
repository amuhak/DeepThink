import os
import re
import httpx

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import flash_client

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")
FIRECRAWL_URL = os.environ.get("FIRECRAWL_URL", "http://firecrawl-api:3002")

MAX_TOOL_ITERATIONS = 10

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
   **PDF WARNING:** When searching for academic papers, prefer HTML versions (e.g., arXiv abstract page, Scholar, conference website) over PDF links. PDFs often lose LaTeX math formatting - even pdf-inspector cannot reconstruct proper equations from font glyphs. Look for URLs like `arxiv.org/abs/...` instead of `arxiv.org/pdf/...`.

4. EXECUTION IS REAL: Do NOT simulate or "thought-block" your tools.
5. NO LOOPING: If you have tried searching/scraping twice with no results and you have no more ideas/options, admit it and output your FINAL summary.

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
    # Add paywall exclusions for academic queries
    academic_keywords = ["paper", "research", "algorithm", "equation", "model", "theorem", "proof", "academic", "journal", "publication"]
    if any(kw in query.lower() for kw in academic_keywords):
        # Exclude major paywall sites
        exclusions = " -site:sciencedirect.com -site:elsevier.com -site:springer.com -site:nature.com -site:ieee.com -site:acm.org"
        query = query + exclusions

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

    # Global memory from previous loops
    failed_urls = state.get("failed_urls", [])
    failed_queries = state.get("failed_queries", [])

    # Debug logging
    debug_log = open(f"/tmp/worker_{worker_id}_debug.log", "w")
    debug_log.write(f"[Worker {worker_id}] START - prompt_type={prompt_type}, prompt_length={len(prompt_text)}\n")

    # Get config from environment
    import os
    flash_llm_url = os.environ.get("FLASH_LLM_URL", "NOT SET")
    flash_model = os.environ.get("FLASH_MODEL", "NOT SET")
    debug_log.write(f"[Worker {worker_id}] FLASH_LLM_URL={flash_llm_url}\n")
    debug_log.write(f"[Worker {worker_id}] FLASH_MODEL={flash_model}\n")

    # Health check: test if FLASH_LLM_URL is reachable
    if flash_llm_url != "NOT SET":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                test_resp = await client.get(flash_llm_url.replace("/v1", "/models"))
                debug_log.write(f"[Worker {worker_id}] Health check status: {test_resp.status_code}\n")
                if test_resp.status_code == 200:
                    debug_log.write(f"[Worker {worker_id}] Health check OK: {test_resp.text[:200]}\n")
                else:
                    debug_log.write(f"[Worker {worker_id}] Health check FAILED: {test_resp.status_code} - {test_resp.text[:200]}\n")
        except Exception as e:
            debug_log.write(f"[Worker {worker_id}] Health check ERROR: {str(e)}\n")
    else:
        debug_log.write(f"[Worker {worker_id}] FLASH_LLM_URL not set, skipping health check\n")

    writer = get_stream_writer()
    writer({"event": "flash_start", "worker": worker_id, "type": prompt_type})

    # Add context about what to avoid
    avoidance_context = ""
    if failed_urls:
        avoidance_context += f"\n\nDO NOT visit these URLs (they failed previously):\n" + "\n".join(failed_urls[:10])
    if failed_queries:
        avoidance_context += f"\n\nDO NOT search for these exact queries (they failed previously):\n" + "\n".join(failed_queries[:10])

    system_with_memory = FLASH_SYSTEM
    if avoidance_context:
        system_with_memory = FLASH_SYSTEM + "\n\n" + avoidance_context

    conversation = [
        {"role": "system", "content": system_with_memory},
        {"role": "user", "content": prompt_text},
    ]
    execution_logs = []
    # Only block last 50 failed queries to allow retry of transient failures
    used_queries = set(failed_queries[-50:]) if failed_queries else set()
    final_response = ""
    worker_timed_out = False
    local_failed_urls = []
    local_failed_queries = set()
    no_tool_prompt_count = 0

    debug_log.write(f"[Worker {worker_id}] Conversation prepared. System length={len(system_with_memory)}, User length={len(prompt_text)}\n")
    debug_log.write(f"[Worker {worker_id}] Calling flash_client.invoke()...\n")

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

        debug_log.write(f"[Worker {worker_id}] LLM response received: timed_out={resp.timed_out}, content_length={len(resp.content) if resp.content else 0}\n")
        debug_log.write(f"[Worker {worker_id}] Content preview: {repr(resp.content[:200]) if resp.content else 'EMPTY'}\n")
        debug_log.write(f"[Worker {worker_id}] Partial preview: {repr(resp.partial[:200]) if resp.partial else 'None'}\n")

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
            writer({"event": "worker_stop", "worker": worker_id, "reason": "FINAL_marker", "iteration": iteration})
            break

        codes = extract_code_blocks(resp.content)
        searches = extract_search_queries(resp.content)
        scrapes = extract_scrape_urls(resp.content)

        if not codes and not searches and not scrapes:
            writer({"event": "worker_stop", "worker": worker_id, "reason": "no_tools", "iteration": iteration})
            # If no final answer yet, prompt once then stop to avoid loops
            if not has_final_answer(final_response) and no_tool_prompt_count < 1:
                conversation.append(
                    {
                        "role": "user",
                        "content": "You haven't provided a FINAL answer. Please either output FINAL: followed by your answer, or continue working by searching/scraping/executing code.",
                    }
                )
                no_tool_prompt_count += 1
                continue
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
            writer({"event": "worker_stop", "worker": worker_id, "reason": "repetitive_tool", "iteration": iteration})
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
                # Only add to failed queries if search returned no results (not transient errors)
                if search_result == "No results found.":
                    local_failed_queries.add(query)
                tool_results.append(
                    f"Search results for '{query}':\n{search_result[:5000]}"
                )

            for url in scrapes:
                writer({"event": "scraping", "worker": worker_id, "url": url})
                scrape_result = await run_scrape(url, worker_id)

                # Track failed URLs (paywalls, errors, empty results)
                failure_indicators = ["error", "blocked", "paywall", "403", "forbidden", "requires authentication", "access denied", "too short"]
                is_failure = any(indicator in scrape_result.lower() for indicator in failure_indicators) or len(scrape_result) < 100
                if is_failure:
                    local_failed_urls.append(url)

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
    # Fallback: if output is empty or just whitespace, add error context
    if not output_text or not output_text.strip():
        output_text = "[WORKER produced no output - likely API failure or timeout]"

    # Log if we hit max iterations without explicit stop
    if not has_final_answer(final_response) and not worker_timed_out:
        writer({"event": "worker_stop", "worker": worker_id, "reason": "max_iterations", "iteration": MAX_TOOL_ITERATIONS - 1})

    writer({"event": "flash_done", "worker": worker_id, "type": prompt_type})

    debug_log.write(f"[Worker {worker_id}] END - output_length={len(output_text)}, timed_out={worker_timed_out}\n")
    debug_log.write(f"[Worker {worker_id}] Output preview: {repr(output_text[:200])}\n")
    debug_log.close()

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
        "failed_urls": local_failed_urls,
        "failed_queries": list(local_failed_queries),
    }
