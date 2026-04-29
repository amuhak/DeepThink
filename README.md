# DeepThink

Local DeepMind-style Deep Think clone using LangGraph multi-agent orchestration with parallel exploration, code execution, and web search.

## Architecture

```
User → OpenAI API → Orchestrator (LangGraph)
                              ├── Pro LLM (advisor_planner / advisor_evaluator)
                              ├── Flash LLM × N (flash_worker via Send() API)
                              ├── Code Sandbox (isolated Python execution)
                              └── SearXNG (local web search)
```

### Nodes

| Node | LLM | Purpose |
|---|---|---|
| `advisor_planner` | Pro | Generates research plan + N balanced prompts (prove/refute) |
| `flash_worker` | Flash | Parallel worker (via `Send()`). Internal tool loop: call LLM → execute code/search → iterate |
| `advisor_evaluator` | Pro | Evaluates all results, decides SOLVED/RETRY/PIVOT |

### Flow

```
START → advisor_planner → [Send() × N] → flash_worker (parallel)
                                              ↓ (converge via operator.add)
                                        advisor_evaluator
                                              ↓
                                     SOLVED → END
                                     RETRY  → flash_worker (with critique)
                                     PIVOT  → advisor_planner (new strategy)
```

## Quick Start

```bash
# 1. Copy and edit environment
cp .env.example .env

# 2. Start all services
docker compose up --build

# 3. Test the API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepthink",
    "messages": [{"role": "user", "content": "Prove that the sum of two even numbers is even"}],
    "stream": true
  }'
```

## API

### `POST /v1/chat/completions`

OpenAI-compatible endpoint.

**Request:**
```json
{
  "model": "deepthink",
  "messages": [{"role": "user", "content": "Your research prompt"}],
  "stream": false,
  "max_loops": 10,
  "num_explorers": 4
}
```

**Response (non-streaming):**
```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "deepthink",
  "choices": [{
    "message": {"role": "assistant", "content": "Final answer..."},
    "finish_reason": "stop"
  }],
  "usage": {"loops": 3, "workers": 4, "status": "SOLVED"}
}
```

**Streaming:** SSE format with `text/event-stream`. Events include planning progress, worker status, code execution, and final answer.

### `GET /v1/models`

Returns available models: `[{"id": "deepthink"}]`

### `GET /health`

Health check endpoint.

## CLI Debug Tool

```bash
# Basic usage
python orchestrator/cli.py "Prove Fermat's Last Theorem for n=4"

# Verbose mode (shows all events and state)
python orchestrator/cli.py "Is 7 a prime number?" --verbose

# Custom config
python orchestrator/cli.py "Explore the Riemann hypothesis" --loops 5 --explorers 6
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PRO_LLM_URL` | `https://llm.amuhak.com/v1` | Pro model endpoint (OpenAI-compatible) |
| `FLASH_LLM_URL` | `https://llm.prnt.ink/v1` | Flash model endpoint |
| `PRO_MODEL` | `unsloth/Qwen3.6-27B` | Pro model name |
| `FLASH_MODEL` | `unsloth/Qwen3.6-27B` | Flash model name |
| `NUM_FLASH_EXPLORERS` | `4` | Number of parallel Flash workers |
| `FLASH_TIMEOUT_SECONDS` | `180` | Timeout per LLM request (seconds) |
| `MAX_LOOPS` | `10` | Maximum iteration loops before forced stop |
| `ORCHESTRATOR_PORT` | `8000` | Host port for the orchestrator API |

## Services

| Service | Port | Description |
|---|---|---|
| `orchestrator` | 8000 | LangGraph agent framework + OpenAI API |
| `searxng` | 8080 | Local web search engine (exposed for testing) |
| `code-sandbox` | internal only | Isolated Python execution (numpy, sympy, scipy, matplotlib, pandas) |

## Design Decisions

- **`Send()` API for parallelism**: LangGraph's native map-reduce pattern. Each Flash worker is a separate graph node instance with individual checkpointing and observability.
- **`operator.add` reducers**: Required for `Send()` workers to append results to shared state without overwriting.
- **Double-ping timeout strategy**: If first LLM call times out, a second attempt is made. Only marked as lost if both fail. Prevents false timeouts on slow llama.cpp servers.
- **Flash worker internal loop**: Each worker iterates up to 5 times — calling Flash LLM, executing code/search tools, feeding results back — until the worker produces a final answer.
- **Streaming via `get_stream_writer()`**: Custom events flow through LangGraph's v2 streaming format, mapped to SSE for the OpenAI API.
- **Sandbox isolation**: `cap_drop: ALL`, 512MB RAM, 1 CPU, no network access, no host ports.

## Directory Structure

```
DeepThink/
├── docker-compose.yml
├── .env.example
├── orchestrator/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py              # FastAPI entrypoint
│   ├── api.py               # OpenAI-compatible endpoints + streaming
│   ├── cli.py               # Debug CLI
│   ├── state.py             # DeepThinkState TypedDict
│   ├── graph.py             # LangGraph graph builder
│   ├── llm_client.py        # Async LLM client with timeout resilience
│   └── nodes/
│       ├── __init__.py
│       ├── advisor_planner.py
│       ├── flash_worker.py
│       └── advisor_evaluator.py
├── code-sandbox/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py
└── searxng/
    └── settings.yml
```
