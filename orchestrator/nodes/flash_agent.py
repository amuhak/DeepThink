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
    
    writer({"event": "state_change", "state": "thinking"})

    conversation = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role") or m.get("role_type", "user")
        content = m.get("content", "")
        conversation.append({"role": role, "content": content})
        
    final_content = ""
    accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    streamed_final_chars_count = 0
    
    max_tool_executions = 9
    tool_executions_count = 0
    
    for iteration in range(15):
        buffered_tokens = []
        
        def on_token(chunk, is_reasoning=False):
            # Always buffer all tokens of this iteration to classify reasoning vs final content
            buffered_tokens.append((chunk, is_reasoning))

        # Filter tools to only include what FlashAgent supports (exclude get_pdf_nexttime)
        if tool_executions_count >= max_tool_executions:
            supported_tools = []
            # Inject a quota exhaustion warning to force the model to synthesize the final answer
            if not any(msg.get("role") == "user" and "quota has been exhausted" in msg.get("content", "") for msg in conversation):
                conversation.append({
                    "role": "user",
                    "content": "SYSTEM WARNING: Your maximum tool execution quota of 9 calls has been exhausted. You cannot call any more tools. You MUST immediately synthesize and write your final answer now using the information gathered so far."
                })
        else:
            supported_tools = [t for t in TOOLS if t["function"]["name"] in ["run_code", "run_search", "run_scrape"]]
            
        resp = await flash_client.invoke(
            conversation,
            tools=supported_tools,
            on_token=on_token,
            temperature=0.6,
        )
        
        # Determine if this iteration is the final answer (no tool calls generated)
        is_final = not resp.tool_calls
        
        if is_final:
            writer({"event": "state_change", "state": "synthesizing"})

        # Flush the buffered tokens with the appropriate is_reasoning flag
        for chunk, client_is_reasoning in buffered_tokens:
            if is_final:
                # Simulate smooth real-time streaming for the final answer
                await asyncio.sleep(0.003)
                final_is_reasoning = client_is_reasoning
            else:
                # Force tool reasoning logs into the thinking accordion
                final_is_reasoning = True
                
            if not final_is_reasoning:
                streamed_final_chars_count += len(chunk.strip())
                
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
            except json.JSONDecodeError as e:
                # Return descriptive syntax error directly to the ReAct context
                res_str = f"ERROR: Invalid JSON format in tool call. {str(e)}. Please ensure arguments are valid JSON."
                writer({"event": "tool_error", "worker": 0, "error": res_str})
                conversation.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": res_str
                })
                continue
                
            # Valid tool call initiated, increment count
            tool_executions_count += 1
            
            res_str = ""
            if name == "run_code":
                code_str = args.get("code") or args.get("script") or args.get("python") or ""
                if not code_str.strip():
                    res_str = "ERROR: Missing required parameter 'code' in run_code. Please call run_code with {'code': 'your_code_string'}"
                else:
                    writer({"event": "code_executing", "worker": 0, "code": code_str})
                    log = await run_code(code_str, worker_id=0)
                    status = "SUCCESS" if log["exit_code"] == 0 else f"FAILED (exit {log['exit_code']})"
                    res_str = f"Code execution {status}:\nstdout:\n{log['stdout']}\nstderr:\n{log['stderr']}"
            elif name == "run_search":
                query = args.get("query") or args.get("query_string") or args.get("q") or ""
                if not query.strip():
                    res_str = "ERROR: Missing required parameter 'query' in run_search. Please call run_search with {'query': 'your_search_query'}"
                else:
                    writer({"event": "searching", "worker": 0, "query": query})
                    res_str = await run_search(query, worker_id=0)
            elif name == "run_scrape":
                url = args.get("url") or args.get("link") or ""
                if not url.strip():
                    res_str = "ERROR: Missing required parameter 'url' in run_scrape. Please call run_scrape with {'url': 'target_url'}"
                else:
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
            
    # Final Synthesis Safeguard: Ensure the user ALWAYS receives a visible final answer.
    # If the ReAct loop ended naturally or due to exhaustion, but we only streamed reasoning
    # tokens (or final_content remains empty), force one last non-tool invocation.
    if streamed_final_chars_count < 10 or not final_content.strip():
        writer({"event": "state_change", "state": "synthesizing"})
        
        conversation.append({
            "role": "user",
            "content": "You have completed your research. Please compile and write your final, complete answer now. Write it directly and clearly, summarizing all findings and answering the user's prompt fully. Do NOT output any internal thoughts or XML tool call tags. Begin your final answer immediately."
        })
        
        buffered_tokens = []
        def on_token_final(chunk, is_reasoning=False):
            buffered_tokens.append((chunk, is_reasoning))
            
        resp = await flash_client.invoke(
            conversation,
            tools=[],
            on_token=on_token_final,
            temperature=0.6,
        )
        
        # Flush the final answer tokens cleanly as final content (is_reasoning=False)
        for chunk, client_is_reasoning in buffered_tokens:
            await asyncio.sleep(0.003)
            writer({
                "event": "token",
                "source": "Think",
                "text": chunk,
                "is_reasoning": False
            })
            
        if resp.content:
            final_content = resp.content
            
        if resp.usage:
            u = resp.usage
            accumulated_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += u.get("completion_tokens", 0)
            accumulated_usage["total_tokens"] += u.get("total_tokens", 0)
            writer({
                "event": "usage",
                "usage": u
            })
            
    return {
        "content": final_content,
        "usage": accumulated_usage
    }
