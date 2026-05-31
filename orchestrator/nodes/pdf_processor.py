import os
import json
import base64
import asyncio
import httpx
import fitz  # PyMuPDF
from langgraph.config import get_stream_writer
from state import DeepThinkState
from llm_client import pro_client


def parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    pages = set()
    if not pages_str:
        return list(range(min(total_pages, 20)))
        
    pages_str = pages_str.strip().lower()
    if pages_str == "all":
        return list(range(total_pages))
    
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start_str, end_str = part.split("-")
                start = max(1, int(start_str.strip()))
                end = min(total_pages, int(end_str.strip()))
                for p in range(start - 1, end):
                    pages.add(p)
            except ValueError:
                pass
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
            except ValueError:
                pass
    if not pages:
        return list(range(min(total_pages, 20)))
    return sorted(list(pages))


async def process_single_pdf(
    pdf_item: dict, 
    idx: int, 
    semaphore: asyncio.Semaphore, 
    writer, 
    debug_log
) -> dict | None:
    url = pdf_item.get("url", "")
    question = pdf_item.get("question", "")
    pages_str = pdf_item.get("pages", "")
    worker_id = pdf_item.get("worker_id", 0)

    debug_log.write(f"[PDF Processor] Starting PDF {idx} - URL: {url}, Question: {question}, Pages: {pages_str}\n")
    writer({"event": "token", "source": "PDF Processor", "text": f"\n- [PDF Processor] Downloading PDF {idx}: {url}...\n"})

    # Step 1: Download PDF bytes safely with a fake User-Agent to avoid blocks
    pdf_bytes = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=45.0, headers=headers) as client:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code == 200:
                pdf_bytes = resp.content
            else:
                debug_log.write(f"[PDF Processor] Download failed for {url} - Status Code: {resp.status_code}\n")
                writer({"event": "token", "source": "PDF Processor", "text": f"\n- [PDF Processor] Download failed for {url} (HTTP {resp.status_code})\n"})
                return None
    except Exception as e:
        debug_log.write(f"[PDF Processor] Download error for {url}: {str(e)}\n")
        writer({"event": "token", "source": "PDF Processor", "text": f"\n- [PDF Processor] Download error for {url}: {str(e)}\n"})
        return None

    # Step 2: Open and render PDF pages in-memory using PyMuPDF (fitz)
    images_b64 = []
    pages_to_render = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_to_render = parse_page_range(pages_str, len(doc))
        debug_log.write(f"[PDF Processor] Rendering pages {pages_to_render} (Total pages: {len(doc)})\n")
        writer({"event": "token", "source": "PDF Processor", "text": f"- [PDF Processor] Rendering pages {sorted([p+1 for p in pages_to_render])} for PDF {idx}...\n"})

        for p_idx in pages_to_render:
            page = doc[p_idx]
            # Use Matrix scale (1.5x) to get clean resolution while avoiding massive file bloat
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("jpeg")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images_b64.append(img_b64)
        doc.close()
    except Exception as e:
        debug_log.write(f"[PDF Processor] Rendering error for {url}: {str(e)}\n")
        writer({"event": "token", "source": "PDF Processor", "text": f"\n- [PDF Processor] Rendering error for {url}: {str(e)}\n"})
        return None

    if not images_b64:
        debug_log.write(f"[PDF Processor] No pages rendered for {url}\n")
        return None

    # Step 3: Run the multimodal vision analysis with the Pro LLM
    debug_log.write(f"[PDF Processor] Invoking Pro LLM for PDF {idx} vision analysis...\n")
    writer({"event": "token", "source": "PDF Processor", "text": f"- [PDF Processor] Analyzing PDF {idx} with Multimodal Pro LLM...\n"})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an elite research assistant. You are provided with pages of a PDF rendered as images. "
                "Carefully extract all useful formulas, LaTeX equations, diagrams, charts, tables, and structured "
                "technical data that directly answer the specific question provided. Omit conversational filler."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"PDF Source URL: {url}\nSpecific Question to Answer: {question}\nPlease analyze the attached page images and extract findings."
                }
            ]
        }
    ]

    for p_idx, b64 in zip(pages_to_render, images_b64):
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    def on_token(chunk, is_reasoning=False):
        # Forward token stream to UI
        writer({"event": "token", "source": f"PDF Processor {idx}", "text": chunk})

    # Acquire semaphore to enforce at most 2 parallel LLM requests
    async with semaphore:
        resp = await pro_client.invoke(
            messages=messages,
            temperature=0.6, # optimal temperature for high-fidelity extraction
            on_token=on_token,
        )

    debug_log.write(f"[PDF Processor] Pro LLM response for PDF {idx} completed: timed_out={resp.timed_out}, content_length={len(resp.content)}\n")
    
    return {
        "worker_id": f"pdf_vision_{idx}",
        "prompt_type": "prove",
        "response": f"FINAL: [High-Fidelity PDF Vision Analysis of {url}]\nQuestion Asked: {question}\n\n{resp.content}",
        "timed_out": resp.timed_out,
        "usage": resp.usage
    }


async def pdf_processor(state: DeepThinkState) -> dict:
    try:
        writer = get_stream_writer()
    except Exception:
        # Fallback dummy writer that prints to stdout/stderr
        def dummy_writer(event_dict):
            text = event_dict.get("text", "")
            if text:
                print(f"[STREAM FALLBACK] {text}", end="", flush=True)
        writer = dummy_writer
    pending = state.get("pending_pdfs", [])
    
    if not pending:
        return {}

    writer({"event": "token", "source": "PDF Processor", "text": f"\n\n[PDF Processor] Initializing vision pipeline for {len(pending)} scheduled PDF links...\n"})
    
    debug_log = open("/tmp/pdf_processor_debug.log", "w")
    debug_log.write(f"[PDF Processor] START - Pending count: {len(pending)}\n")

    # Enforce maximum of 2 parallel LLM requests to balance speed and VRAM safety
    semaphore = asyncio.Semaphore(2)
    tasks = [
        process_single_pdf(item, i, semaphore, writer, debug_log)
        for i, item in enumerate(pending)
    ]
    
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    debug_log.write(f"[PDF Processor] Completed. Valid results: {len(valid_results)}\n")
    debug_log.close()

    if not valid_results:
        return {"pending_pdfs": []} # Clear queue even if failed to avoid loops

    # Aggregate token usage stats
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in valid_results:
        usage = r.pop("usage", None)
        if usage:
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

    # Return the findings directly into flash_outputs so Evaluator & Synthesizer receive them seamlessly!
    return {
        "flash_outputs": valid_results,
        "pending_pdfs": [], # Clear queue to avoid re-processing in subsequent loops
        "usage": total_usage
    }
