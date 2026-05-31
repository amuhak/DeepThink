import os
import re
import json
import httpx

from langgraph.config import get_stream_writer

from state import DeepThinkState
from llm_client import flash_client

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")
FIRECRAWL_URL = os.environ.get("FIRECRAWL_URL", "http://firecrawl-api:3002")

MAX_TOOL_ITERATIONS = 10

FLASH_SYSTEM = """You are a high-performance research worker. You have direct access to a Python sandbox, a Web Search engine, and a URL Scraper.

TOOL RULES & INSTRUCTIONS:
1. PYTHON (`run_code`): Execute python code inside a secure isolated sandbox container. RESTRICTION: Do NOT write code to scrape websites (e.g., requests, bs4, selenium). Use run_scrape instead.
   **SANDBOX TOOLKIT:** The sandbox is equipped with standard high-performance Python libraries pre-installed and available:
   - Numerical & Tensors: `numpy`, `torch` (PyTorch CPU)
   - Machine Learning & Stats: `scipy`, `scikit-learn`
   - Data & Graphs: `pandas`, `pyarrow`, `networkx`
   - Symbolic & Rendering: `sympy`, `matplotlib`, `PIL` (Pillow)
   - Parsing: `beautifulsoup4`, `lxml`
2. SEARCH (`run_search`): Search the web using SearXNG.
3. SCRAPE (`run_scrape`): Read the full markdown content of a specific URL (arXiv, GitHub, documentation, website). Use this to extract mathematical formulas, code snippets, and in-depth details.
   **PDF WARNING:** When searching for academic papers, prefer HTML abstract pages (e.g. arXiv abstract page) over raw PDF links.
4. NO LOOPING: If you have tried searching/scraping twice with no results and have no more ideas, stop calling tools and provide your FINAL summary.

OUTPUT FORMAT:
- First, briefly think step-by-step and call appropriate tools.
- Your final answer must start with "FINAL:" on its own line.
- **SOURCE ATTRIBUTION:** For every key fact, you MUST state where you found it (e.g., "Source: [GitHub URL]").
- Always extract mathematical formulas and code snippets.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute python code inside a secure isolated sandbox container. RESTRICTION: Do NOT write code to scrape websites (e.g. requests, bs4, selenium). Use run_scrape instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The exact python code to run in the sandbox."
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_search",
            "description": "Search the web using a search engine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_scrape",
            "description": "Read the full markdown content of a specific URL (arXiv abstract, GitHub, documentation, website). Use this when you find a promising technical link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute URL to scrape."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pdf_nexttime",
            "description": "Schedule a PDF URL to be analyzed in high-fidelity (including LaTeX math, charts, and equations) using a vision-capable LLM. Use this when you find a promising technical PDF link that has structural details/charts/formulas that scrape tools fail to extract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute URL of the PDF to analyze."
                    },
                    "question": {
                        "type": "string",
                        "description": "The specific technical question or details you want to extract from this PDF."
                    },
                    "pages": {
                        "type": "string",
                        "description": "Optional page range to analyze (e.g. '1-5' or '1,3,5'). If not specified, the first 20 pages are analyzed."
                    }
                },
                "required": ["url", "question"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


def has_final_answer(text: str) -> bool:
    if not text:
        return False
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
        print(
            f"[Worker {worker_id}] Search '{query}': {len(results)} results from {data.get('engines', [])}"
        )
        snippets = []
        for r in results:
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{FIRECRAWL_URL}/v1/scrape",
                json={"url": url, "formats": ["markdown"]},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("data", {}).get("markdown", "")[:50000]
    except Exception:
        pass

    # Fallback to r.jina.ai
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(jina_url)
        return resp.text[:50000]
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
    pro_llm_url = os.environ.get("PRO_LLM_URL", "NOT SET")
    pro_model = os.environ.get("PRO_MODEL", "NOT SET")
    debug_log.write(f"[Worker {worker_id}] PRO_LLM_URL={pro_llm_url}\n")
    debug_log.write(f"[Worker {worker_id}] PRO_MODEL={pro_model}\n")

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
    local_pending_pdfs = []
    no_tool_prompt_count = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    debug_log.write(f"[Worker {worker_id}] Conversation prepared. System length={len(system_with_memory)}, User length={len(prompt_text)}\n")

    def on_token(chunk, is_reasoning=False):
        writer({"event": "token", "source": f"Worker {worker_id}", "text": chunk})

    for iteration in range(MAX_TOOL_ITERATIONS):
        resp = await flash_client.invoke(
            conversation,
            tools=TOOLS,
            on_token=on_token,
            temperature=0.6,
        )

        if resp.usage:
            total_usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += resp.usage.get("total_tokens", 0)

        debug_log.write(f"[Worker {worker_id}] LLM response received: timed_out={resp.timed_out}, content_length={len(resp.content) if resp.content else 0}, tool_calls={len(resp.tool_calls)}\n")
        debug_log.write(f"[Worker {worker_id}] LLM content: {resp.content}\n")
        debug_log.write(f"[Worker {worker_id}] LLM tool calls: {resp.tool_calls}\n")

        if resp.timed_out:
            writer(
                {"event": "flash_timeout", "worker": worker_id, "iteration": iteration}
            )
            final_response = (
                f"[TIMEOUT on iteration {iteration}] Partial: {resp.partial}"
            )
            worker_timed_out = True
            break

        final_response = resp.content or ""
        
        # Append assistant's response to conversation in standard OpenAI tool calling flow
        assistant_msg = {"role": "assistant"}
        if resp.content:
            assistant_msg["content"] = resp.content
        if resp.tool_calls:
            assistant_msg["tool_calls"] = resp.tool_calls
        conversation.append(assistant_msg)

        if has_final_answer(resp.content):
            writer({"event": "worker_stop", "worker": worker_id, "reason": "FINAL_marker", "iteration": iteration})
            break

        if not resp.tool_calls:
            writer({"event": "worker_stop", "worker": worker_id, "reason": "no_tools", "iteration": iteration})
            # If no final answer yet, prompt once then stop to avoid loops
            if not has_final_answer(final_response) and no_tool_prompt_count < 1:
                conversation.append(
                    {
                        "role": "user",
                        "content": "You haven't provided a FINAL answer. Please either output FINAL: followed by your answer, or continue working by calling tools.",
                    }
                )
                no_tool_prompt_count += 1
                continue
            break

        # Check for repetitive tool calls
        repetitive = False
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            arguments_str = tc["function"]["arguments"]
            try:
                args = json.loads(arguments_str)
            except Exception:
                args = {}
            val = args.get("query", "") or args.get("url", "")
            if val and val in used_queries:
                repetitive = True
                break
            if val:
                used_queries.add(val)

        if repetitive:
            writer({"event": "worker_stop", "worker": worker_id, "reason": "repetitive_tool", "iteration": iteration})
            for tc in resp.tool_calls:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": "ERROR: Repetitive tool call detected. You are repeating a query or URL that already failed or yielded no new info. Please STOP and provide your FINAL summary based on what you already found."
                })
            continue

        # Execute each tool call
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            arguments_str = tc["function"]["arguments"]
            call_id = tc["id"]

            try:
                args = json.loads(arguments_str)
            except Exception as e:
                args = {}

            result_str = ""
            if name == "run_code":
                code = args.get("code", "")
                writer({"event": "code_executing", "worker": worker_id, "iteration": iteration})
                log = await run_code(code, worker_id)
                execution_logs.append(log)
                status = "SUCCESS" if log["exit_code"] == 0 else f"FAILED (exit {log['exit_code']})"
                result_str = f"Code execution {status}:\nstdout:\n{log['stdout']}\nstderr:\n{log['stderr']}"

            elif name == "run_search":
                query = args.get("query", "")
                writer({"event": "searching", "worker": worker_id, "query": query})
                search_result = await run_search(query, worker_id)
                execution_logs.append({
                    "worker_id": worker_id,
                    "code": f"SEARCH: {query}",
                    "stdout": search_result[:10000],
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                })
                if search_result == "No results found.":
                    local_failed_queries.add(query)
                result_str = f"Search results for '{query}':\n{search_result[:5000]}"

            elif name == "run_scrape":
                url = args.get("url", "")
                writer({"event": "scraping", "worker": worker_id, "url": url})
                scrape_result = await run_scrape(url, worker_id)
                
                failure_indicators = ["error", "blocked", "paywall", "403", "forbidden", "requires authentication", "access denied", "too short"]
                is_failure = any(indicator in scrape_result.lower() for indicator in failure_indicators) or len(scrape_result) < 100
                if is_failure:
                    local_failed_urls.append(url)

                execution_logs.append({
                    "worker_id": worker_id,
                    "code": f"SCRAPE: {url}",
                    "stdout": scrape_result[:2000],
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                })
                result_str = f"Scraped content from {url}:\n{scrape_result}"
            elif name == "get_pdf_nexttime":
                url = args.get("url", "")
                question = args.get("question", "")
                pages = args.get("pages", "")
                writer({"event": "token", "source": f"Worker {worker_id}", "text": f"\n- [Worker {worker_id}] Scheduled PDF link for vision analysis: {url}...\n"})
                local_pending_pdfs.append({
                    "url": url,
                    "question": question,
                    "pages": pages,
                    "worker_id": worker_id
                })
                result_str = f"Successfully scheduled PDF ({url}) for high-fidelity vision processing. It will be rendered and analyzed by the multimodal Pro LLM immediately after this research turn completes."
            else:
                result_str = f"ERROR: Unknown tool '{name}'."

            conversation.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result_str
            })

    output_text = final_response if has_final_answer(final_response) else final_response
    if not output_text or not output_text.strip():
        output_text = "[WORKER produced no output - likely API failure or timeout]"

    if not has_final_answer(final_response) and not worker_timed_out:
        writer({"event": "worker_stop", "worker": worker_id, "reason": "max_iterations", "iteration": MAX_TOOL_ITERATIONS - 1})

    writer({"event": "flash_done", "worker": worker_id, "type": prompt_type})

    debug_log.write(f"[Worker {worker_id}] END - output_length={len(output_text)}, timed_out={worker_timed_out}\n")
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
        "usage": total_usage,
        "pending_pdfs": local_pending_pdfs,
    }
