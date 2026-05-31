import time
import json
import uuid

from fastapi import HTTPException
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "think"
    messages: list[ChatMessage]
    stream: bool = False
    max_loops: int | None = None
    num_explorers: int | None = None
    stream_options: StreamOptions | None = None


def extract_user_prompt(messages: list[ChatMessage]) -> str:
    last_user = ""
    for msg in messages:
        if msg.role == "user":
            last_user = msg.content
    return last_user or messages[-1].content if messages else ""


def build_sse_chunk(
    chunk_id: str, content: str, finish_reason: str | None = None, usage: dict | None = None
) -> str:
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepthink",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        data["usage"] = usage
    if finish_reason:
        data["choices"][0]["delta"] = {}
    return f"data: {json.dumps(data)}\n\n"


async def run_streaming(
    graph, messages: list[ChatMessage], config_overrides: dict | None = None, stream_options: StreamOptions | None = None
):
    user_prompt = extract_user_prompt(messages)
    chunk_id = str(uuid.uuid4())
    yield build_sse_chunk(chunk_id, "<thinking>\n")
    thinking_open = True
    last_source = None
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    config = {"recursion_limit": 100}
    if config_overrides:
        config.update(config_overrides)

    try:
        async for part in graph.astream(
            {
                "user_prompt": user_prompt,
                "chat_history": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "status": "RUNNING",
                "loop_count": 0,
                "flash_outputs": [],
                "execution_logs": [],
                "evaluation_history": [],
                "flash_prompts": [],
                "current_plan": "",
                "final_answer": "",
                "pending_pdfs": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            stream_mode=["updates", "custom"],
            version="v2",
            config=config,
        ):
            ptype = part.get("type", "")

            if ptype == "custom":
                data = part.get("data", {})
                event = data.get("event", "")

                if event == "token":
                    source = data.get("source", "?")
                    text = data.get("text", "")
                    is_reasoning = data.get("is_reasoning", False)

                    if source == "Synthesizer":
                        if is_reasoning:
                            if thinking_open:
                                if last_source != source:
                                    yield build_sse_chunk(chunk_id, f"\n\n[{source} Thinking] ")
                                    last_source = source
                                yield build_sse_chunk(chunk_id, text)
                        else:
                            if thinking_open:
                                yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
                                thinking_open = False
                            yield build_sse_chunk(chunk_id, text)
                    elif not thinking_open:
                        yield build_sse_chunk(chunk_id, text)
                    elif source in ["Planner", "Evaluator", "PDF Processor"]:
                        if last_source != source:
                            yield build_sse_chunk(chunk_id, f"\n\n[{source} Thinking] ")
                            last_source = source
                        yield build_sse_chunk(chunk_id, text)
                elif event == "planning":
                    pass
                elif event == "flash_start":
                    pass
                elif event == "code_executing":
                    wid = data.get("worker", 0)
                    yield build_sse_chunk(
                        chunk_id, f"\n- [Worker {wid}] Executing code...\n"
                    )
                elif event == "searching":
                    wid = data.get("worker", 0)
                    query = data.get("query", "")
                    yield build_sse_chunk(
                        chunk_id, f"\n- [Worker {wid}] Searching: {query}\n"
                    )
                elif event == "scraping":
                    wid = data.get("worker", 0)
                    url = data.get("url", "")
                    yield build_sse_chunk(
                        chunk_id, f"\n- [Worker {wid}] Scraping: {url}\n"
                    )
                elif event == "flash_done":
                    pass
                elif event == "evaluating":
                    pass
                elif event == "decision":
                    status = data.get("status", "?")
                    loop = data.get("loop", 0)
                    if status != "SOLVED":
                        yield build_sse_chunk(
                            chunk_id, f"\n- [Loop {loop} Ended] Status: {status} (Next step: {data.get('reason', '')[:60]}...)\n"
                        )
                elif event == "synthesizing":
                    pass

            elif ptype == "updates":
                for node_name, node_update in part.get("data", {}).items():
                    if node_update and isinstance(node_update, dict) and "usage" in node_update and node_update["usage"]:
                        u = node_update["usage"]
                        total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += u.get("completion_tokens", 0)
                        total_usage["total_tokens"] += u.get("total_tokens", 0)

    except Exception as e:
        if thinking_open:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n\n</thinking>\n\n")
            thinking_open = False
        else:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n")

    if thinking_open:
        yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
        thinking_open = False

    yield build_sse_chunk(chunk_id, "", finish_reason="stop", usage=total_usage)


async def run_blocking(
    graph, messages: list[ChatMessage], config_overrides: dict | None = None
) -> dict:
    user_prompt = extract_user_prompt(messages)
    config = {"recursion_limit": 100}
    if config_overrides:
        config.update(config_overrides)

    result = await graph.ainvoke(
        {
            "user_prompt": user_prompt,
            "chat_history": [{"role": m.role, "content": m.content} for m in messages],
            "status": "RUNNING",
            "loop_count": 0,
            "flash_outputs": [],
            "execution_logs": [],
            "evaluation_history": [],
            "flash_prompts": [],
            "current_plan": "",
            "final_answer": "",
            "pending_pdfs": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        config=config,
    )

    final_answer = result.get("final_answer", "")
    if not final_answer and result.get("status") == "SOLVED":
        outputs = result.get("flash_outputs", [])
        final_answer = outputs[0]["response"] if outputs else "No answer generated."
    elif not final_answer:
        final_answer = f"[Completed with status: {result.get('status', 'UNKNOWN')}] No final answer synthesized."

    return {
        "status": result.get("status", "UNKNOWN"),
        "final_answer": final_answer,
        "loops": result.get("loop_count", 0),
        "plan": result.get("current_plan", ""),
        "worker_count": len(result.get("flash_outputs", [])),
        "usage": result.get("usage"),
    }


async def run_flash_agent_stream(
    messages: list[ChatMessage], stream_options: StreamOptions | None = None
):
    import asyncio
    import uuid
    chunk_id = str(uuid.uuid4())
    yield build_sse_chunk(chunk_id, "<thinking>\n")
    thinking_open = True
    last_source = None
    
    from nodes.flash_agent import run_flash_agent
    
    # Event synchronization queue
    queue = asyncio.Queue()
    
    def writer(event_dict):
        queue.put_nowait(event_dict)
        
    # Launch FlashAgent as a background task
    agent_task = asyncio.create_task(run_flash_agent(
        [{"role": m.role, "content": m.content} for m in messages],
        writer
    ))
    
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    while not agent_task.done() or not queue.empty():
        try:
            if not queue.empty():
                data = queue.get_nowait()
            else:
                data = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
            
        event = data.get("event", "")
        if event == "token":
            source = data.get("source", "?")
            text = data.get("text", "")
            is_reasoning = data.get("is_reasoning", False)
            
            if is_reasoning:
                if thinking_open:
                    yield build_sse_chunk(chunk_id, text)
            else:
                if thinking_open:
                    yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
                    thinking_open = False
                yield build_sse_chunk(chunk_id, text)
        elif event == "usage":
            u = data.get("usage", {})
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)
            total_usage["total_tokens"] += u.get("total_tokens", 0)
        elif event == "code_executing":
            code = data.get("code", "")
            yield build_sse_chunk(chunk_id, f"\n- [Think] Executing code:\n```python\n{code}\n```\n")
        elif event == "searching":
            query = data.get("query", "")
            yield build_sse_chunk(chunk_id, f"\n- [Think] Searching: {query}\n")
        elif event == "scraping":
            url = data.get("url", "")
            yield build_sse_chunk(chunk_id, f"\n- [Think] Scraping: {url}\n")
            
    try:
        agent_result = await agent_task
        final_answer = agent_result.get("content", "")
        # Merge stats
        u = agent_result.get("usage", {})
        total_usage["prompt_tokens"] = max(total_usage["prompt_tokens"], u.get("prompt_tokens", 0))
        total_usage["completion_tokens"] = max(total_usage["completion_tokens"], u.get("completion_tokens", 0))
        total_usage["total_tokens"] = max(total_usage["total_tokens"], u.get("total_tokens", 0))
    except Exception as e:
        if thinking_open:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n\n</thinking>\n\n")
            thinking_open = False
        else:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n")
            
    if thinking_open:
        yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
        thinking_open = False
        
    yield build_sse_chunk(chunk_id, "", finish_reason="stop", usage=total_usage)


async def run_flash_agent_blocking(messages: list[ChatMessage]) -> dict:
    from nodes.flash_agent import run_flash_agent
    def dummy_writer(event_dict):
        pass
    result = await run_flash_agent(
        [{"role": m.role, "content": m.content} for m in messages],
        dummy_writer
    )
    return {
        "status": "SOLVED",
        "final_answer": result["content"],
        "loops": 1,
        "plan": "Single-call FlashAgent Execution",
        "worker_count": 1,
        "usage": result["usage"]
    }

