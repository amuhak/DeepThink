# DeepThink

Local DeepMind-style Deep Think clone using LangGraph multi-agent orchestration with parallel exploration, code execution, deep web scraping, and authoritative synthesis.

## Architecture

```
User → OpenAI API → Orchestrator (LangGraph)
                              ├── PRO LLM (advisor_planner / advisor_evaluator / advisor_synthesizer)
                              ├── FLASH LLM × N (flash_worker via Send() API)
                              ├── Code Sandbox (isolated Python execution)
                              ├── Firecrawl (headless browser scraping) + Jina fallback
                              └── SearXNG (local web search)
```

### Nodes

| Node | LLM | Purpose |
|---|---|---|
| `advisor_planner` | PRO | Strategic planner. Generates N balanced prompts (prove/refute) and instructs workers on tool use. |
| `flash_worker` | FLASH | High-speed researcher. Parallel execution via `Send()`. Uses `search`, `scrape` (Firecrawl + Jina), and `python`. Each worker runs up to 5 tool iterations internally. |
| `advisor_evaluator` | PRO | Senior Technical Lead. Critiques worker output, detects loops, and decides SOLVED/RETRY/PIVOT. |
| `advisor_synthesizer`| PRO | Final Report Writer. Polishes raw research into an elite, LaTeX-heavy technical report. |

### Flow

```
START → advisor_planner → [Send() × N] → flash_worker (parallel)
                                              ↓ (converge via operator.add)
                                        advisor_evaluator
                                              ↓
                                     SOLVED → advisor_synthesizer → END
                                     RETRY  → flash_worker (with critique)
                                     PIVOT  → advisor_planner (new strategy)
```

### Worker Stop Reasons

Workers can stop due to:
- `FINAL_marker`: Worker output contains `FINAL:` prefix
- `no_tools`: No code/search/scrape blocks detected in response
- `repetitive_tool`: Same query already tried
- `max_iterations`: Hit 5 tool iteration limit
- `timeout`: FLASH model timed out

## Quick Start

```bash
# 1. Copy and edit environment
cp .env.example .env

# 2. Start all services
docker compose up --build -d

# 3. Test the API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepthink",
    "messages": [{"role": "user", "content": "Explain the PolarQuant algorithm and its mathematical foundation"}],
    "stream": true
  }'
```

## Development Workflow

To ensure stability and safety, the orchestrator code is copied into the container during the build process rather than live-mounted.

**To apply code changes:**
```bash
docker compose build orchestrator
docker compose up -d --force-recreate orchestrator
```

## API

### `POST /v1/chat/completions`

OpenAI-compatible endpoint. Supports real-time thought streaming.

**Request:**
```json
{
  "model": "deepthink",
  "messages": [{"role": "user", "content": "Your research prompt"}],
  "stream": true
}
```

**Streaming Events:**
- `planning`: The advisor is designing the research strategy.
- `worker_stop`: Worker stopped with reason (FINAL_marker, no_tools, repetitive_tool, max_iterations)
- `searching` / `scraping` / `code_executing`: Real-time tool usage tracking.
- `flash_timeout`: Worker timed out during tool execution
- `decision`: Evaluator decision (SOLVED/RETRY/PIVOT + loop count)
- `synthesizing`: The final report is being polished.
- `token`: Content chunks for the final answer.

**Direct LLM Access:**
- `model: "flash"` - Call FLASH directly (bypasses graph)
- `model: "pro"` - Call PRO directly (bypasses graph)
- `model: "deepthink"` - Full LangGraph orchestration

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PRO_LLM_URL` | `http://host.docker.internal:8001/v1` | PRO model endpoint (OpenAI-compatible) |
| `FLASH_LLM_URL` | - | FLASH model endpoint |
| `PRO_MODEL` | - | Reasoning model (e.g., Qwen3.6-27B) |
| `FLASH_MODEL` | - | Speed model (e.g., NVIDIA-Nemotron-3-Nano-Omni-30B) |
| `NUM_FLASH_EXPLORERS` | `4` | Number of parallel Flash workers |
| `MAX_LOOPS` | `3` | Maximum iteration loops before forced synthesis |
| `FLASH_TIMEOUT_SECONDS` | `120` | Timeout for FLASH model calls |

## Services

| Service | Port | Description |
|---|---|---|
| `orchestrator` | 8000 | LangGraph agent framework + OpenAI API |
| `searxng` | 8080 | Local web search engine |
| `firecrawl-api`| 3002 | Headless browser API for deep scraping |
| `code-sandbox` | internal | Isolated Python execution (numpy, sympy, scipy, pandas) |
| `openwebui` | 3000 | (Optional) Modern UI for interacting with DeepThink |

## Design Decisions

- **Silent Thinking Filter**: PRO/FLASH models may output `reasoning_content` with internal thinking. This is merged into the main content stream for display.
- **Firecrawl + Jina Fallback**: Workers use local Firecrawl for scraping, falling back to `r.jina.ai` if it fails. Note: Both struggle with PDF extraction - prefer HTML sources.
- **Exhaustive Reporting**: The system produces high-fidelity technical reports with LaTeX and code snippets, rather than simple summaries.
- **Safety**: Each worker is restricted to 5 tool iterations and is automatically terminated if it detects a query loop.
- **JSON Resilience**: Planner and evaluator use custom parsers that handle markdown code fences, truncated JSON, and invalid escape characters.
- **Three-Retry Timeout**: LLM calls retry up to 3 times on 524/504 gateway errors before marking as failed.

## Known Limitations

- **PDF Extraction**: Firecrawl and Jina.ai convert PDFs to plain text but lose LaTeX formatting. Workers should prefer HTML versions of papers (arXiv, Scholar, etc.) over PDFs.
- **FLASH Speed**: The FLASH endpoint may be slower than PRO. Worker tool calls and search results are truncated to manage payload size.