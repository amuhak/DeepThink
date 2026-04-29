# DeepThink

Local DeepMind-style Deep Think clone using LangGraph multi-agent orchestration with parallel exploration, code execution, deep web scraping, and authoritative synthesis.

## Architecture

```
User → OpenAI API → Orchestrator (LangGraph)
                              ├── Pro LLM (advisor_planner / advisor_evaluator / advisor_synthesizer)
                              ├── Flash LLM × N (flash_worker via Send() API)
                              ├── Code Sandbox (isolated Python execution)
                              ├── Firecrawl (headless browser scraping)
                              └── SearXNG (local web search)
```

### Nodes

| Node | LLM | Purpose |
|---|---|---|
| `advisor_planner` | Pro | Strategic planner. Generates N balanced prompts (prove/refute) and instructs workers on tool use. |
| `flash_worker` | Flash | High-speed researcher. Parallel execution via `Send()`. Uses `search`, `scrape` (Firecrawl), and `python`. |
| `advisor_evaluator` | Pro | Senior Technical Lead. Critiques worker output, detects loops, and decides if the problem is SOLVED. |
| `advisor_synthesizer`| Pro | Final Report Writer. Polishes raw research into an elite, LaTeX-heavy technical report. |

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
docker compose up --build -d orchestrator
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
- `flash_start`: Workers have begun parallel exploration.
- `searching` / `scraping` / `code_executing`: Real-time tool usage tracking.
- `evaluating`: The Senior Lead is reviewing the evidence.
- `synthesizing`: The final report is being polished.
- `token`: Content chunks for the final answer.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PRO_LLM_URL` | - | Pro model endpoint (OpenAI-compatible) |
| `FLASH_LLM_URL` | - | Flash model endpoint |
| `PRO_MODEL` | - | Reasoning model (e.g., Qwen-32B-Instruct) |
| `FLASH_MODEL` | - | Speed model (e.g., Llama-3.1-8B) |
| `NUM_FLASH_EXPLORERS` | `4` | Number of parallel Flash workers |
| `MAX_LOOPS` | `10` | Maximum iteration loops before forced stop |

## Services

| Service | Port | Description |
|---|---|---|
| `orchestrator` | 8000 | LangGraph agent framework + OpenAI API |
| `searxng` | 8080 | Local web search engine |
| `firecrawl-api`| 3002 | Headless browser API for deep scraping |
| `code-sandbox` | internal | Isolated Python execution (numpy, sympy, scipy, pandas) |
| `openwebui` | 3000 | (Optional) Modern UI for interacting with DeepThink |

## Design Decisions

- **Silent Thinking Filter**: Internal monologues (reasoning tokens) from Pro models are filtered from the final SSE stream to provide a clean UX.
- **Firecrawl + Jina Fallback**: Workers use a local Firecrawl stack for scraping, falling back to `r.jina.ai` if the local service times out.
- **Exhaustive Reporting**: The system is tuned to produce high-fidelity technical reports with LaTeX and code snippets, rather than simple summaries.
- **Safety**: Each worker is restricted to 5 tool iterations and is automatically terminated if it detects a query loop.
