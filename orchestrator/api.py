import time
import json
import uuid
import asyncio
from collections import OrderedDict

from fastapi import HTTPException
from pydantic import BaseModel


class DeepThinkQueue:
    def __init__(self):
        self._queue = OrderedDict()  # request_id -> asyncio.Event
        self._current_running = None
        self._lock = asyncio.Lock()

    def register(self, request_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._queue[request_id] = event
        # If queue is empty and nothing is running, trigger immediately
        if len(self._queue) == 1 and not self._current_running:
            event.set()
        return event

    def get_position(self, request_id: str) -> int:
        if request_id not in self._queue:
            return 0
        keys = list(self._queue.keys())
        return keys.index(request_id) + 1

    async def acquire(self, request_id: str):
        event = self._queue.get(request_id)
        if not event:
            return
        await event.wait()
        async with self._lock:
            self._current_running = request_id

    def release(self, request_id: str):
        if request_id in self._queue:
            del self._queue[request_id]
        if self._current_running == request_id:
            self._current_running = None
        # Wake up the next waiting request in FIFO order
        if self._queue:
            next_id = next(iter(self._queue))
            self._queue[next_id].set()

deepthink_queue = DeepThinkQueue()


from typing import Any

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


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
            if isinstance(msg.content, str):
                last_user = msg.content
            elif isinstance(msg.content, list):
                text_parts = [part.get("text", "") for part in msg.content if isinstance(part, dict) and part.get("type") == "text"]
                last_user = "\n".join(text_parts)
    if not last_user and messages:
        last_msg = messages[-1].content
        if isinstance(last_msg, str):
            last_user = last_msg
        elif isinstance(last_msg, list):
            text_parts = [part.get("text", "") for part in last_msg if isinstance(part, dict) and part.get("type") == "text"]
            last_user = "\n".join(text_parts)
    return last_user


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
    request_id = str(uuid.uuid4())
    yield build_sse_chunk(chunk_id, "<thinking>\n")
    thinking_open = True
    last_source = None
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    config = {"recursion_limit": 100}
    if config_overrides:
        config.update(config_overrides)

    event = deepthink_queue.register(request_id)

    try:
        last_pos = -1
        while True:
            if event.is_set():
                break
            pos = deepthink_queue.get_position(request_id)
            if pos != last_pos:
                yield build_sse_chunk(chunk_id, f"\n- [Queue] Position {pos} in queue. Waiting for prior deepthink tasks to complete...\n")
                last_pos = pos
            yield ": keep-alive\n\n"
            await asyncio.sleep(2.0)

        await deepthink_queue.acquire(request_id)
        yield build_sse_chunk(chunk_id, f"\n- [Queue] Active! Starting DeepThink research loop...\n")

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
                    event_type = data.get("event", "")

                    if event_type == "token":
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
                        else:
                            # Send keep-alive SSE comments for worker tokens to prevent connection timeouts
                            yield ": keep-alive\n\n"
                    elif event_type == "planning":
                        pass
                    elif event_type == "flash_start":
                        pass
                    elif event_type == "code_executing":
                        wid = data.get("worker", 0)
                        yield build_sse_chunk(
                            chunk_id, f"\n- [Worker {wid}] Executing code...\n"
                        )
                    elif event_type == "searching":
                        wid = data.get("worker", 0)
                        query = data.get("query", "")
                        yield build_sse_chunk(
                            chunk_id, f"\n- [Worker {wid}] Searching: {query}\n"
                        )
                    elif event_type == "scraping":
                        wid = data.get("worker", 0)
                        url = data.get("url", "")
                        yield build_sse_chunk(
                            chunk_id, f"\n- [Worker {wid}] Scraping: {url}\n"
                        )
                    elif event_type == "flash_done":
                        pass
                    elif event_type == "evaluating":
                        pass
                    elif event_type == "decision":
                        status = data.get("status", "?")
                        loop = data.get("loop", 0)
                        if status != "SOLVED":
                            yield build_sse_chunk(
                                chunk_id, f"\n- [Loop {loop} Ended] Status: {status} (Next step: {data.get('reason', '')[:60]}...)\n"
                            )
                    elif event_type == "synthesizing":
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
    finally:
        deepthink_queue.release(request_id)

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

    request_id = str(uuid.uuid4())
    event = deepthink_queue.register(request_id)
    try:
        await event.wait()
        await deepthink_queue.acquire(request_id)
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
    finally:
        deepthink_queue.release(request_id)

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
        if event == "state_change":
            pass  # Handled dynamically by token's is_reasoning flag
        elif event == "token":
            text = data.get("text", "")
            is_reasoning = data.get("is_reasoning", False)
            if is_reasoning:
                if not thinking_open:
                    yield build_sse_chunk(chunk_id, "\n<thinking>\n")
                    thinking_open = True
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

