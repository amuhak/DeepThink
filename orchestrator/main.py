import os
import json
import time
import uuid
import asyncio
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from graph import build_graph
from api import (
    ChatCompletionRequest,
    extract_user_prompt,
    run_streaming,
    run_blocking,
    build_sse_chunk,
    run_flash_agent_stream,
    run_flash_agent_blocking,
)
from llm_client import pro_client, flash_client

dependency_health = {"status": "ok", "details": {}}

async def monitor_dependencies():
    searxng_url = os.environ.get("SEARXNG_URL", "http://searxng:8080")
    sandbox_url = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")
    firecrawl_url = os.environ.get("FIRECRAWL_URL", "http://firecrawl-api:3002")
    pro_llm_url = os.environ.get("PRO_LLM_URL", "https://llm.amuhak.com/v1")

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            details = {}
            status = "ok"
            try:
                res = await client.get(f"{searxng_url}/healthz")
                details["searxng"] = res.status_code == 200
                if res.status_code != 200: status = "down"
            except Exception:
                details["searxng"] = False
                status = "down"
            
            try:
                res = await client.get(f"{sandbox_url}/health")
                details["sandbox"] = res.status_code == 200
                if res.status_code != 200: status = "down"
            except Exception:
                details["sandbox"] = False
                status = "down"
            
            try:
                res = await client.get(f"{firecrawl_url}/test")
                details["firecrawl"] = True
            except Exception:
                details["firecrawl"] = False
                status = "down"

            try:
                res = await client.get(f"{pro_llm_url}/models")
                details["pro_llm"] = res.status_code == 200
                if res.status_code != 200: status = "down"
            except Exception:
                details["pro_llm"] = False
                status = "down"
            
            dependency_health["status"] = status
            dependency_health["details"] = details
            
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitor_dependencies())
    yield
    task.cancel()

app = FastAPI(title="DeepThink Orchestrator", lifespan=lifespan)
graph = build_graph()


@app.get("/v1/models")
async def list_models():
    if dependency_health["status"] == "down":
        return {
            "object": "list",
            "data": []
        }
        
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "think",
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
    if dependency_health["status"] == "down":
        return JSONResponse(
            status_code=503, 
            content={
                "error": {
                    "message": f"One or more dependent services are down. Details: {dependency_health['details']}",
                    "type": "server_error",
                    "code": "service_unavailable"
                }
            }
        )

    user_prompt = extract_user_prompt(req.messages)

    if not user_prompt:
        return JSONResponse(
            status_code=400, content={"error": "No user message provided"}
        )

    config_overrides = {}
    if req.max_loops:
        os.environ["MAX_LOOPS"] = str(req.max_loops)
    if req.num_explorers:
        # Cap num_explorers at a maximum of 4
        os.environ["NUM_FLASH_EXPLORERS"] = str(min(req.num_explorers, 4))

    if req.model == "think":
        if req.stream:
            return StreamingResponse(
                run_flash_agent_stream(req.messages, req.stream_options),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await run_flash_agent_blocking(req.messages)
            chunk_id = str(uuid.uuid4())
            return JSONResponse(
                content={
                    "id": chunk_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "think",
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
                    "usage": result.get("usage") or {
                        "loops": result["loops"],
                        "workers": result["worker_count"],
                        "status": result["status"],
                    },
                }
            )
    else:
        if req.stream:
            return StreamingResponse(
                run_streaming(graph, req.messages, config_overrides, req.stream_options),
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
                    "usage": result.get("usage") or {
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
    return {
        "status": dependency_health["status"], 
        "service": "deepthink-orchestrator",
        "dependencies": dependency_health["details"]
    }
