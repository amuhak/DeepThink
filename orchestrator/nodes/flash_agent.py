import json
import asyncio
from state import DeepThinkState
from llm_client import flash_client
from nodes.flash_worker import TOOLS, run_code, run_search, run_scrape

async def run_flash_agent(messages: list[dict], writer) -> str:
    """
    Executes a single-call tool-using agent (FlashAgent) for simple tasks.
    Runs a ReAct loop with a maximum of 10 tool-use iterations.
    Streams reasoning and content chunks dynamically using the Synthesizer source for UX alignment.
    """
    system_prompt = (
        "You are an elite fast-path assistant named 'think'. You have direct access to a Python sandbox, "
        "a Web Search engine, and a URL Scraper. Solve the user's task directly. "
        "Keep your reasoning concise and call tools immediately when needed. "
        "The sandbox is equipped with standard high-performance Python libraries pre-installed and available:\n"
        "- Numerical & Tensors: numpy, torch (PyTorch CPU)\n"
        "- Machine Learning & Stats: scipy, scikit-learn\n"
        "- Data & Graphs: pandas, pyarrow, networkx\n"
        "- Symbolic & Rendering: sympy, matplotlib, PIL (Pillow)\n"
        "- Parsing: beautifulsoup4, lxml\n\n"
        "OUTPUT FORMAT:\n"
        "- Summarize your final answer clearly.\n"
        "- Cite your sources using complete, untruncated Markdown links (e.g., [Title](URL)) for any facts found via search/scrape. Do NOT truncate, shorten, or omit the URLs under any circumstances.\n"
        "- STRICTLY FORBID META-COMMENTARY & PREAMBLES: Absolutely do NOT output any conversational filler, meta-commentary, transition phrases, or self-summaries (e.g., never say 'I now have a comprehensive understanding...', 'I have all the information...', 'Here is the summary:', or 'I will summarize the key points...'). Begin your final answer immediately, cleanly, and directly with the facts, code, or explanation."
    )
    
    conversation = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role") or m.get("role_type", "user")
        content = m.get("content", "")
        conversation.append({"role": role, "content": content})
        
    final_content = ""
    accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for iteration in range(10):
        buffered_tokens = []
        
        def on_token(chunk, is_reasoning=False):
            # Always buffer all tokens of this iteration to classify reasoning vs final content
            buffered_tokens.append((chunk, is_reasoning))

        # Filter tools to only include what FlashAgent supports (exclude get_pdf_nexttime)
        supported_tools = [t for t in TOOLS if t["function"]["name"] in ["run_code", "run_search", "run_scrape"]]
        
        if iteration == 9:
            supported_tools = []
            
        resp = await flash_client.invoke(
            conversation,
            tools=supported_tools,
            on_token=on_token,
            temperature=0.6,
        )
        
        # Determine if this iteration is the final answer (no tool calls generated)
        is_final = not resp.tool_calls
        
        # Flush the buffered tokens with the appropriate is_reasoning flag
        for chunk, client_is_reasoning in buffered_tokens:
            if is_final:
                # Simulate smooth real-time streaming for the final answer
                await asyncio.sleep(0.003)
                final_is_reasoning = client_is_reasoning
            else:
                # Force tool reasoning logs into the thinking accordion
                final_is_reasoning = True
                
            writer({
                "event": "token",
                "source": "Think",
                "text": chunk,
                "is_reasoning": final_is_reasoning
            })
        
        if resp.usage:
            u = resp.usage
            accumulated_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += u.get("completion_tokens", 0)
            accumulated_usage["total_tokens"] += u.get("total_tokens", 0)
            writer({
                "event": "usage",
                "usage": u
            })
        
        # Build assistant message
        assistant_msg = {"role": "assistant"}
        if resp.content:
            assistant_msg["content"] = resp.content
            final_content = resp.content
        if resp.tool_calls:
            assistant_msg["tool_calls"] = resp.tool_calls
        conversation.append(assistant_msg)
        
        if not resp.tool_calls:
            break
            
        # Execute each tool call sequentially
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            arguments_str = tc["function"]["arguments"]
            call_id = tc["id"]
            
            try:
                args = json.loads(arguments_str)
            except Exception:
                args = {}
                
            res_str = ""
            if name == "run_code":
                code_str = args.get("code", "")
                writer({"event": "code_executing", "worker": 0, "code": code_str})
                log = await run_code(code_str, worker_id=0)
                status = "SUCCESS" if log["exit_code"] == 0 else f"FAILED (exit {log['exit_code']})"
                res_str = f"Code execution {status}:\nstdout:\n{log['stdout']}\nstderr:\n{log['stderr']}"
            elif name == "run_search":
                query = args.get("query", "")
                writer({"event": "searching", "worker": 0, "query": query})
                res_str = await run_search(query, worker_id=0)
            elif name == "run_scrape":
                url = args.get("url", "")
                writer({"event": "scraping", "worker": 0, "url": url})
                res_str = await run_scrape(url, worker_id=0)
            else:
                res_str = f"ERROR: Unknown tool '{name}'."
                
            conversation.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": res_str
            })
            
    return {
        "content": final_content,
        "usage": accumulated_usage
    }
