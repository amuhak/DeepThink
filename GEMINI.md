# DeepThink

Local DeepMind-style Deep Think clone using LangGraph multi-agent orchestration with parallel exploration, code execution, deep web scraping, and authoritative synthesis.

## Project Overview

DeepThink is a multi-agent system designed for deep research and complex problem-solving. It leverages a hierarchical "Advisor-Worker" architecture to decompose user prompts into strategic research plans, execute parallel investigations using specialized tools, and synthesize high-quality technical reports.

### Architecture
- **Orchestrator (LangGraph):** Manages the state and flow between different nodes.
- **Advisor Planner (PRO LLM):** Breaks down the initial request into research prompts for parallel workers.
- **Flash Workers (FLASH LLM):** Execute research in parallel. Each worker has access to:
    - **Web Search:** Powered by a local SearXNG instance.
    - **Web Scraping:** Powered by Firecrawl with a Jina fallback.
    - **Python Sandbox:** An isolated environment for executing code.
- **Advisor Evaluator (PRO LLM):** Critiques worker outputs and determines if the research is complete or needs further iteration (SOLVED/RETRY/PIVOT).
- **Advisor Synthesizer (PRO LLM):** Compiles the final findings into a LaTeX-heavy technical report.

### Key Technologies
- **Python (FastAPI):** Main framework for the orchestrator and API endpoints.
- **LangGraph:** Framework for building stateful, multi-agent workflows.
- **Docker:** Containerization for the orchestrator, sandbox, search, and scraping services.
- **SearXNG:** Local metasearch engine.
- **Firecrawl:** Headless browser API for scraping.

## Building and Running

The project is designed to run within a Docker environment.

### Prerequisites
- Docker and Docker Compose
- Environment variables configured in a `.env` file (see `.env.example`).

### Key Commands
- **Start all services:**
  ```bash
  docker compose up --build -d
  ```
- **Apply code changes to the orchestrator:**
  ```bash
  docker compose build orchestrator
  docker compose up -d --force-recreate orchestrator
  ```
- **Run tests:**
  There are numerous test scripts in the root directory (e.g., `test_orchestrator.py`, `test_flash.py`). These can typically be run using `python <test_file>.py`.
- **Check worker logs:**
  ```bash
  docker exec deep-think-orchestrator cat /tmp/worker_0_debug.log
  ```

## Development Conventions

- **State Management:** The orchestrator uses `DeepThinkState` (defined in `orchestrator/state.py`) to manage global state across loops.
- **Tool Usage:** Workers use specialized blocks (```python, ```search, ```scrape) to interact with the environment.
- **LLM Interaction:** The `LLMClient` (in `orchestrator/llm_client.py`) handles communication with OpenAI-compatible endpoints, supporting streaming and JSON modes.
- **Resilience:** The system includes custom JSON parsing to handle LaTeX and malformed LLM outputs, as well as global memory to avoid redundant failures.
- **Security:** Code execution is performed in an isolated `code-sandbox` container with restricted capabilities.

## Directory Structure
- `orchestrator/`: Core logic, FastAPI server, and LangGraph nodes.
- `code-sandbox/`: Isolated Python execution environment.
- `searxng/`: Configuration for the local search engine.
- `openwebui/`: Dockerfile for the optional user interface.
- `images/`: Brand assets.
- Root: Docker Compose and various test scripts.
