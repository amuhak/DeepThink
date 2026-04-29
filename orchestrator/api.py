import time
import json
import uuid

from fastapi import HTTPException
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "flash"
    messages: list[ChatMessage]
    stream: bool = False
    max_loops: int | None = None
    num_explorers: int | None = None


def extract_user_prompt(messages: list[ChatMessage]) -> str:
    last_user = ""
    for msg in messages:
        if msg.role == "user":
            last_user = msg.content
    return last_user or messages[-1].content if messages else ""


def build_sse_chunk(
    chunk_id: str, content: str, finish_reason: str | None = None
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
    if finish_reason:
        data["choices"][0]["delta"] = {}
    return f"data: {json.dumps(data)}\n\n"


async def run_streaming(graph, user_prompt: str, config_overrides: dict | None = None):
    chunk_id = str(uuid.uuid4())
    yield build_sse_chunk(chunk_id, "<thinking>\n")
    thinking_open = True
    last_source = None

    config = {"recursion_limit": 100}
    if config_overrides:
        config.update(config_overrides)

    try:
        async for part in graph.astream(
            {
                "user_prompt": user_prompt,
                "status": "RUNNING",
                "loop_count": 0,
                "flash_outputs": [],
                "execution_logs": [],
                "evaluation_history": [],
                "flash_prompts": [],
                "current_plan": "",
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
                    if thinking_open:
                        if last_source != source:
                            yield build_sse_chunk(chunk_id, f"\n\n[{source}] {text}")
                            last_source = source
                        else:
                            yield build_sse_chunk(chunk_id, text)
                    else:
                        # Stream directly (e.g. from synthesizer)
                        yield build_sse_chunk(chunk_id, text)
                elif event == "planning":
                    last_source = None
                    yield build_sse_chunk(
                        chunk_id, "\n[Planning research strategy...]\n"
                    )
                elif event == "plan_generated":
                    plan = data.get("plan", "")
                    yield build_sse_chunk(chunk_id, f"\n**Plan:** {plan}\n\n")
                elif event == "flash_start":
                    last_source = None
                    wid = data.get("worker", 0)
                    wtype = data.get("type", "?")
                    yield build_sse_chunk(
                        chunk_id, f"\n[Worker {wid} ({wtype}) started...]\n"
                    )
                elif event == "code_executing":
                    last_source = None
                    wid = data.get("worker", 0)
                    yield build_sse_chunk(
                        chunk_id, f"\n[Worker {wid} executing code...]\n"
                    )
                elif event == "searching":
                    last_source = None
                    wid = data.get("worker", 0)
                    query = data.get("query", "")
                    yield build_sse_chunk(
                        chunk_id, f"\n[Worker {wid} searching: {query}...]\n"
                    )
                elif event == "flash_done":
                    last_source = None
                    wid = data.get("worker", 0)
                    wtype = data.get("type", "?")
                    yield build_sse_chunk(
                        chunk_id, f"\n[Worker {wid} ({wtype}) complete]\n"
                    )
                elif event == "evaluating":
                    last_source = None
                    yield build_sse_chunk(chunk_id, "\n[Evaluating results...]\n")
                elif event == "decision":
                    last_source = None
                    status = data.get("status", "?")
                    reason = data.get("reason", "")
                    yield build_sse_chunk(
                        chunk_id, f"\n[Decision: {status}] {reason[:150]}\n"
                    )
                elif event == "flash_timeout":
                    last_source = None
                    wid = data.get("worker", 0)
                    yield build_sse_chunk(chunk_id, f"\n[Worker {wid} timed out]\n")
                elif event == "synthesizing":
                    last_source = None
                    if thinking_open:
                        yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
                        thinking_open = False
                    yield build_sse_chunk(chunk_id, "### Analysis Complete\n\n")

            elif ptype == "updates":
                # Final answer is now handled by synthesizer tokens, 
                # we don't need to yield it from evaluator updates anymore.
                pass

    except Exception as e:
        if thinking_open:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n\n</thinking>\n\n")
            thinking_open = False
        else:
            yield build_sse_chunk(chunk_id, f"\n[Error: {str(e)}]\n")

    if thinking_open:
        yield build_sse_chunk(chunk_id, "\n</thinking>\n\n")
        thinking_open = False

    yield build_sse_chunk(chunk_id, "", finish_reason="stop")


async def run_blocking(
    graph, user_prompt: str, config_overrides: dict | None = None
) -> dict:
    config = {"recursion_limit": 100}
    if config_overrides:
        config.update(config_overrides)

    result = await graph.ainvoke(
        {
            "user_prompt": user_prompt,
            "status": "RUNNING",
            "loop_count": 0,
            "flash_outputs": [],
            "execution_logs": [],
            "evaluation_history": [],
            "flash_prompts": [],
            "current_plan": "",
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
    }
