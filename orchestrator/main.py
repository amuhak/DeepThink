import os
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from graph import build_graph
from api import (
    ChatCompletionRequest,
    extract_user_prompt,
    run_streaming,
    run_blocking,
    build_sse_chunk,
)
from llm_client import pro_client, flash_client

app = FastAPI(title="DeepThink Orchestrator")
graph = build_graph()


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "flash",
                "object": "model",
                "created": now,
                "owned_by": "deepthink",
            },
            {
                "id": "pro",
                "object": "model",
                "created": now,
                "owned_by": "deepthink",
            },
            {
                "id": "deepthink",
                "object": "model",
                "created": now,
                "owned_by": "deepthink",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    user_prompt = extract_user_prompt(req.messages)

    if not user_prompt:
        return JSONResponse(
            status_code=400, content={"error": "No user message provided"}
        )

    config_overrides = {}
    if req.max_loops:
        os.environ["MAX_LOOPS"] = str(req.max_loops)
    if req.num_explorers:
        os.environ["NUM_FLASH_EXPLORERS"] = str(req.num_explorers)

    if req.stream:
        return StreamingResponse(
            run_streaming(graph, req.messages, config_overrides),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await run_blocking(graph, req.messages, config_overrides)
        chunk_id = str(uuid.uuid4())
        return JSONResponse(
            content={
                "id": chunk_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepthink",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["final_answer"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "loops": result["loops"],
                    "workers": result["worker_count"],
                    "status": result["status"],
                },
            }
        )


@app.post("/v1/completions")
async def completions(req: ChatCompletionRequest):
    return await chat_completions(req)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deepthink-orchestrator"}
