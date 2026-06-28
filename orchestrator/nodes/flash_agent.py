import json
import re
import asyncio
from state import DeepThinkState
from llm_client import flash_client
from nodes.flash_worker import TOOLS, run_code, run_search, run_scrape


def _extract_answer_from_content(raw_content: str) -> tuple[str, str]:
    """
    Parse raw LLM content to separate reasoning from the actual answer.
    
    Qwen3 models emit <think>...</think> blocks, but sometimes continue
    reasoning in plain text after </think> before writing the real answer.
    They may also attempt tool calls via <tool_call> XML in that post-think
    reasoning section. All such content must be classified as reasoning.
    
    Strategy:
    1. Find the last </think> tag to establish the primary boundary.
    2. In the post-</think> text, find any <tool_call> blocks — all text
       up to and including the last tool call block is still reasoning.
    3. Only text after all think/tool_call blocks is the actual answer.
    4. Strip all XML tags from reasoning; strip tool_call blocks from answer.
    
    Returns:
        (reasoning_text, answer_text) — both with tags stripped.
    """
    if not raw_content:
        return ("", "")
    
    # Find the last </think> boundary in the RAW content (before any stripping)
    last_think_close = raw_content.rfind("</think>")
    
    if last_think_close == -1:
        # No </think> found
        if "<think>" in raw_content:
            # Unclosed think block — everything is reasoning
            stripped = re.sub(r"</?think>", "", raw_content)
            stripped = re.sub(r"<tool_call>.*?</tool_call>", "", stripped, flags=re.DOTALL)
            return (stripped.strip(), "")
        # No think blocks at all — check for tool_call blocks
        if "<tool_call>" in raw_content:
            # Has tool calls but no think blocks — find the last tool call end
            last_tc_end = _find_last_toolcall_end(raw_content)
            if last_tc_end > 0:
                reasoning_part = raw_content[:last_tc_end]
                answer_part = raw_content[last_tc_end:]
                reasoning_clean = re.sub(r"<tool_call>.*?</tool_call>", "", reasoning_part, flags=re.DOTALL).strip()
                return (reasoning_clean, answer_part.strip())
                
        # Fallback for "Thinking:" / "Answer:" format
        match = re.search(r"^(.*?)(?:\n\s*)?Answer:(.*)", raw_content, re.DOTALL | re.IGNORECASE)
        if match and "thinking:" in raw_content.lower():
            reasoning = match.group(1).strip()
            if reasoning.lower().startswith("thinking:"):
                reasoning = reasoning[9:].strip()
            answer = match.group(2).strip()
            return (reasoning, answer)
            
        return ("", raw_content)
    
    # Split at the </think> boundary
    reasoning_before = raw_content[:last_think_close + len("</think>")]
    after_think = raw_content[last_think_close + len("</think>"):]
    
    # In the post-</think> section, check for <tool_call> blocks.
    # Any text up to and including the last tool call block is still reasoning.
    last_tc_end = _find_last_toolcall_end(after_think)
    
    if last_tc_end > 0:
        # There are tool call attempts after </think> — classify text around them as reasoning
        additional_reasoning = after_think[:last_tc_end]
        answer_raw = after_think[last_tc_end:]
    else:
        additional_reasoning = ""
        answer_raw = after_think
    
    # Clean up reasoning: merge think blocks + any additional post-think reasoning
    reasoning_clean = re.sub(r"</?think>", "", reasoning_before)
    if additional_reasoning:
        additional_clean = re.sub(r"<tool_call>.*?</tool_call>", "", additional_reasoning, flags=re.DOTALL)
        reasoning_clean += "\n" + additional_clean
    reasoning_clean = reasoning_clean.strip()
    
    # Clean up answer: strip any residual tags
    answer_clean = re.sub(r"<tool_call>.*?</tool_call>", "", answer_raw, flags=re.DOTALL).strip()
    
    return (reasoning_clean, answer_clean)


def _find_last_toolcall_end(text: str) -> int:
    """Find the character position after the last </tool_call> in text. Returns 0 if none found."""
    last_pos = 0
    end_tag = "</tool_call>"
    idx = text.find(end_tag)
    while idx != -1:
        last_pos = idx + len(end_tag)
        idx = text.find(end_tag, last_pos)
    return last_pos

def _get_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""

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
        "- If you need to reason or think step-by-step before answering, you MUST wrap your entire thought process inside `<think>` and `</think>` XML tags. Do not use 'Thinking:' or other formats. Only use `<think>...</think>`.\n"
        "- Summarize your final answer clearly outside of the think tags.\n"
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
            if not any(msg.get("role") == "user" and "quota has been exhausted" in _get_text_from_content(msg.get("content", "")) for msg in conversation):
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
        if is_final:
            # FINAL ITERATION: Don't trust the stream filter's is_reasoning classification.
            # Qwen3 models often continue reasoning in plain text after </think>,
            # which the filter incorrectly marks as is_reasoning=False.
            # Instead, re-parse resp.content to find the true answer boundary.
            reasoning_text, answer_text = _extract_answer_from_content(resp.content or "")
            
            if reasoning_text:
                writer({
                    "event": "token",
                    "source": "Think",
                    "text": reasoning_text,
                    "is_reasoning": True
                })
            
            if answer_text:
                # Stream the actual answer with paced typing
                for i in range(0, len(answer_text), 4):
                    chunk = answer_text[i:i+4]
                    await asyncio.sleep(0.003)
                    writer({
                        "event": "token",
                        "source": "Think",
                        "text": chunk,
                        "is_reasoning": False
                    })
                streamed_final_chars_count += len(answer_text.strip())
        else:
            # NON-FINAL: Force all tokens into the thinking accordion
            for chunk, _client_is_reasoning in buffered_tokens:
                writer({
                    "event": "token",
                    "source": "Think",
                    "text": chunk,
                    "is_reasoning": True
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
            if is_final:
                # Store the clean answer as final_content (not raw content with <think> tags)
                _, clean_answer = _extract_answer_from_content(resp.content)
                final_content = clean_answer or resp.content
            else:
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
        
        # Flush the final answer tokens using the same boundary detection
        reasoning_text, answer_text = _extract_answer_from_content(resp.content or "")
        
        # Fallback safeguard: if the safeguard failed to generate a separate answer,
        # treat the entire content (or reasoning text) as the visible answer.
        if not answer_text.strip():
            answer_text = reasoning_text or resp.content or ""
            reasoning_text = ""
            
        if reasoning_text:
            writer({
                "event": "token",
                "source": "Think",
                "text": reasoning_text,
                "is_reasoning": True
            })
        
        if answer_text:
            for i in range(0, len(answer_text), 4):
                chunk = answer_text[i:i+4]
                await asyncio.sleep(0.003)
                writer({
                    "event": "token",
                    "source": "Think",
                    "text": chunk,
                    "is_reasoning": False
                })
            
        if resp.content:
            _, clean_answer = _extract_answer_from_content(resp.content)
            final_content = clean_answer or resp.content
            
        if resp.usage:
            u = resp.usage
            accumulated_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += u.get("completion_tokens", 0)
            accumulated_usage["total_tokens"] += u.get("total_tokens", 0)
            writer({
                "event": "usage",
                "usage": u
            })
            
    import sys
    print("\n" + "=" * 80, flush=True)
    print("[DEBUG] RAW API RESPONSE (final_content) FOR THINK AGENT:", flush=True)
    print(repr(final_content), flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()
            
    return {
        "content": final_content,
        "usage": accumulated_usage
    }
