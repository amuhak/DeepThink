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
|------|-----|---------|
| `advisor_planner` | PRO | Strategic planner. Generates N balanced prompts (prove/refute) and instructs workers on tool use. |
| `flash_worker` | FLASH | High-speed researcher. Parallel execution via `Send()`. Uses `search`, `scrape` (Firecrawl + Jina), and `python`. Each worker runs up to 5 tool iterations internally. Features global memory to avoid repeating failed URLs/queries. |
| `advisor_evaluator` | PRO | Senior Technical Lead. Critiques worker output, detects loops, and decides SOLVED/RETRY/PIVOT. Includes LaTeX-aware JSON parsing. |
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
- `no_tools`: No code/search/scrape blocks detected in response (now prompts worker to continue instead of silent break)
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

**Debug Mode:**
Enable LLM debug logging by adding to your `.env`:
```
DEBUG_LLM=1
```
This logs SSE events, content length, and HTTP status codes to stdout.

**Worker Debug Logs:**
Workers write debug info to `/tmp/worker_N_debug.log` (where N is the worker ID). Check these logs after runs:
```bash
docker exec deep-think-orchestrator cat /tmp/worker_0_debug.log
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
|----------|---------|-------------|
| `PRO_LLM_URL` | `http://host.docker.internal:8001/v1` | PRO model endpoint (OpenAI-compatible) |
| `FLASH_LLM_URL` | - | FLASH model endpoint |
| `FASTCHAT_LLM_URL` | `http://100.82.53.76:8000` | FastChat model endpoint (used by Open WebUI) |
| `PRO_MODEL` | - | Reasoning model (e.g., unsloth/Qwen3.6-27B) |
| `FLASH_MODEL` | - | Speed model (e.g., unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B) |
| `FASTCHAT_MODEL` | `fastChat` | FastChat model name |
| `NUM_FLASH_EXPLORERS` | `4` | Number of parallel Flash workers |
| `MAX_LOOPS` | `10` | Maximum iteration loops before forced synthesis |
| `FLASH_TIMEOUT_SECONDS` | `120` | Timeout for FLASH model calls |
| `DEBUG_LLM` | `0` | Enable LLM debug logging (set to `1`) |

## Services

| Service | Port | Description |
|---------|------|-------------|
| `orchestrator` | 8000 | LangGraph agent framework + OpenAI API |
| `searxng` | 8080 | Local web search engine |
| `firecrawl-api`| 3002 | Headless browser API for deep scraping |
| `code-sandbox` | internal | Isolated Python execution (numpy, sympy, scipy, pandas) |
| `openwebui` | 3000 | (Optional) Modern UI for interacting with DeepThink. Connects to orchestrator, FastChat, PRO, and FLASH endpoints. |

## Design Decisions

- **Silent Thinking Filter**: PRO/FLASH models may output `reasoning_content` with internal thinking. This is merged into the main content stream for display.
- **Firecrawl + Jina Fallback**: Workers use local Firecrawl for scraping, falling back to `r.jina.ai` if it fails. Both now use PDF Rust extraction (`PDF_RUST_EXTRACT_ENABLE=true`), but workers are instructed to prefer HTML sources (arXiv abs/ vs pdf/) for better LaTeX preservation.
- **Exhaustive Reporting**: The system produces high-fidelity technical reports with LaTeX and code snippets, rather than simple summaries.
- **Safety**: Each worker is restricted to 5 tool iterations and is automatically terminated if it detects a query loop.
- **JSON Resilience**: Planner and evaluator use custom parsers that handle markdown code fences, truncated JSON, invalid escape characters, and LaTeX expressions (e.g., `\frac`, `\sum`, `\alpha`).
- **Three-Retry Timeout**: LLM calls retry up to 3 times on 524/504 gateway errors before marking as failed.
- **Global Memory**: Workers track failed URLs and search queries across loops. The state includes `failed_urls` and `failed_queries` (using `operator.add` reducer), which are passed to subsequent workers via system prompt to avoid repeating failed attempts.
- **Paywall Blacklist**: Academic search queries automatically exclude major paywall sites (`-site:sciencedirect.com -site:elsevier.com -site:springer.com -site:nature.com -site:ieee.com -site:acm.org`) to improve content accessibility.
- **Improved Error Handling**: LLM client now returns `[WORKER ERROR: ...]` or `[TIMEOUT]` in content instead of empty strings, allowing the evaluator to detect and report API failures properly.
- **Worker Output Fallback**: If a worker produces no output (empty string), the system now adds `[WORKER produced no output - likely API failure or timeout]` to surface the issue.

## Known Limitations

- **PDF Extraction**: Firecrawl and Jina.ai convert PDFs to plain text but lose LaTeX formatting. Workers should prefer HTML versions of papers (arXiv, Scholar, etc.) over PDFs.
- **FLASH Speed**: The FLASH endpoint may be slower than PRO. Worker tool calls and search results are truncated to manage payload size.
- **Worker Debugging**: If workers return empty outputs, check `/tmp/worker_N_debug.log` inside the container and enable `DEBUG_LLM=1` in `.env`.
- **LaTeX in JSON**: Despite improved parsing, complex LaTeX expressions in evaluator JSON output may still cause parse failures. The system will retry with evaluator hints.
