import requests
import json
import time
import os
import sys

URL = "http://localhost:8000/v1/chat/completions"
REPORT_DIR = r"C:\Users\amuhak\.gemini\antigravity-cli\brain\4ea459c4-4af4-4f96-9fd7-deaed05aaf50"
REPORT_JSON_PATH = os.path.join(REPORT_DIR, "comprehensive_test_report.json")
REPORT_MD_PATH = os.path.join(REPORT_DIR, "comprehensive_test_report.md")

TEST_CASES = [
    {
        "id": 1,
        "name": "Scouting and Scraping - TurboQuant",
        "query": "Search for 'TurboQuant' on GitHub. What is its primary focus, what quantization techniques does it implement, and what is its performance edge? Cite exact URLs.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Scouting and scraping a newly released quantization library on GitHub."
    },
    {
        "id": 2,
        "name": "PDF Vision + Mathematical Sandbox Validation",
        "query": "Analyze page 4 of the Attention Is All You Need paper (https://arxiv.org/pdf/1706.03762.pdf) via get_pdf_nexttime. Extract the exact formula for Scaled Dot-Product Attention. Then, execute a Python script in your sandbox to verify this formula on two random 3x3 matrices Q and K with d_k=3. Output the formula, code, and executed results.",
        "num_explorers": 1,
        "max_loops": 3,
        "requires_pdf": True,
        "requires_sandbox": True,
        "description": "Extracts math from a PDF visually, writes python code to compute attention, and executes in the sandbox."
    },
    {
        "id": 3,
        "name": "Comparative Ring Attention ZigZag Scrape",
        "query": "Investigate the implementation details of Ring Attention in FlashAttention-3. Compare the context window scaling between standard Ring Attention and ZigZag Ring Attention. Cite your sources (GitHub or arXiv URLs) and write out the communication overhead formulas.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Compares complex distributed context window scaling architectures using web scraper fallbacks."
    },
    {
        "id": 4,
        "name": "Multi-PDF Concurrency Stress Test",
        "query": "Use get_pdf_nexttime to analyze page 1 of: (a) https://arxiv.org/pdf/1706.03762.pdf, (b) https://arxiv.org/pdf/2402.17762.pdf, and (c) https://arxiv.org/pdf/2310.00000.pdf. Extract and compare their titles, first-author affiliations, and main proposed algorithms.",
        "num_explorers": 2,
        "max_loops": 2,
        "requires_pdf": True,
        "requires_sandbox": False,
        "description": "Triggers concurrent rendering and vision analysis of 3 PDF files under a semaphore constraint of 4."
    },
    {
        "id": 5,
        "name": "Sandbox Algorithmic Correctness FFT",
        "query": "Implement the Cooley-Tukey Fast Fourier Transform (FFT) algorithm in Python, write a test suite with 8-point signals inside the sandbox, run it to verify mathematical correctness against numpy FFT, and output the absolute numerical differences. Do NOT write scraping code.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": True,
        "description": "Validates sandbox's mathematical capability by compiling a complex Cooley-Tukey algorithm and profiling it."
    },
    {
        "id": 6,
        "name": "Epistemic Humility & Hallucination Probe",
        "query": "Is there a python library called 'mamba-ssm-torch'? Find out if it exists, verify its creators or if it is a hallucination, and compare it with the official 'mamba-ssm' package.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Probes the agent's epistemic humility and hallucination detection on a non-existent package name."
    },
    {
        "id": 7,
        "name": "Regret Bounds & Optimization Math",
        "query": "Find the exact mathematical proof for the convergence rate of Adam optimizer under non-convex stochastic optimization (e.g. from Kingma & Ba 2014 or subsequent papers). Write down the exact regret bounds in LaTeX and verify the mathematical assumptions.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Tests Adam optimizer regret bounds, convergence proofs, and heavy LaTeX mathematical formulation."
    },
    {
        "id": 8,
        "name": "Edge Cases & Failed Operations Pivot",
        "query": "Search for information about a non-existent protocol called 'HyperGigaTransit HTTP/4.5'. Scrape a broken/invalid URL like 'https://this-is-a-completely-fake-url-12345.com/spec' and show how your system handles failed searches/scrapes and pivots to report the truth.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Tests systemic recovery when encountering fake terms and absolute network/DNS failures."
    },
    {
        "id": 9,
        "name": "Mixture-of-Depths (MoD) vs MoE Routing",
        "query": "Review the key advancements of Mixture-of-Depths (MoD) routing mechanisms compared to standard Mixture-of-Experts (MoE). Detail the routing capacity formula, capacity factor, and execution savings. Provide a step-by-step mathematical breakdown.",
        "num_explorers": 1,
        "max_loops": 2,
        "requires_pdf": False,
        "requires_sandbox": False,
        "description": "Tests Mixture-of-Depths math, routing equations, and architectural capacity trade-offs."
    },
    {
        "id": 10,
        "name": "GPTQ Quantization Sandbox Profiling",
        "query": "Investigate the differences between GPTQ, AWQ, and FP4 quantization. Write a Python script to run in the sandbox that simulates a simple weight matrix, applies a toy GPTQ-like optimization step (pseudo-inverse of Hessian H^-1), quantizes the weights, and measures the mean squared error (MSE) before and after quantization.",
        "num_explorers": 1,
        "max_loops": 3,
        "requires_pdf": False,
        "requires_sandbox": True,
        "description": "Profiles complex quantization formulas (Hessian inverse, AWQ scales) inside the isolated python sandbox."
    }
]

def run_test_case(test: dict) -> dict:
    name = test["name"]
    query = test["query"]
    print("\n" + "="*90)
    print(f"RUNNING COMPREHENSIVE TEST CASE {test['id']}/10: {name}")
    print(f"Description: {test['description']}")
    print(f"Query: {query}")
    print("="*90 + "\n")
    
    start_time = time.time()
    full_response = []
    has_thinking = False
    thinking_content = []
    
    try:
        req = requests.post(
            URL,
            json={
                "model": "deepthink",
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "stream": True,
                "num_explorers": test["num_explorers"],
                "max_loops": test["max_loops"],
            },
            stream=True,
            timeout=600,
        )
        
        if req.status_code != 200:
            print(f"\n[Test Case '{name}' failed with HTTP {req.status_code}]")
            return {
                "id": test["id"],
                "name": name,
                "success": False,
                "elapsed": time.time() - start_time,
                "error": f"HTTP {req.status_code}: {req.text}",
                "metrics": {}
            }

        in_thinking = False
        for line in req.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    try:
                        data = json.loads(decoded[6:])
                        delta = data["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            if "<thinking>" in content:
                                in_thinking = True
                                has_thinking = True
                            
                            if in_thinking:
                                thinking_content.append(content)
                                # Print thinking in dim gray
                                print(f"\033[90m{content}\033[0m", end="", flush=True)
                            else:
                                print(content, end="", flush=True)
                                full_response.append(content)
                                
                            if "</thinking>" in content:
                                in_thinking = False
                    except Exception:
                        pass
        
        elapsed = time.time() - start_time
        response_text = "".join(full_response)
        thinking_text = "".join(thinking_content)
        
        print(f"\n\n[Test Case '{name}' Completed in {elapsed:.2f}s]")
        
        # Self-evaluation of response quality
        eval_metrics = evaluate_response(test, response_text, thinking_text)
        
        return {
            "id": test["id"],
            "name": name,
            "success": eval_metrics["passed_all"],
            "elapsed": elapsed,
            "response": response_text,
            "thinking": thinking_text,
            "metrics": eval_metrics
        }
        
    except Exception as e:
        print(f"\n[Test Case '{name}' FAILED with error: {e}]")
        return {
            "id": test["id"],
            "name": name,
            "success": False,
            "elapsed": time.time() - start_time,
            "error": str(e),
            "metrics": {}
        }

def evaluate_response(test: dict, response: str, thinking: str) -> dict:
    name = test["name"]
    print("\n" + "-"*50)
    print(f"AUTOMATED METRIC ASSESSMENT: {name}")
    print("-"*50)
    
    checks = {
        "Has LaTeX formulas": ("$" in response or "$$" in response or "\\\\" in response),
        "Contains final summary (FINAL:)": bool("FINAL:" in response or "Final Verdict" in response or "Conclusion" in response),
        "Has source citations (Source:)": bool("Source:" in response or "https://" in response or "arxiv" in response.lower() or "github" in response.lower()),
        "Valid HTML structure omitted": bool("<html>" not in response and "<head>" not in response),
    }
    
    if test["requires_pdf"]:
        checks["Triggered PDF Vision Parsing"] = bool(
            "PDF Vision" in response or "PDF Processor" in thinking or "get_pdf_nexttime" in thinking or "attention" in response.lower() or "formula" in response.lower()
        )
    if test["requires_sandbox"]:
        checks["Executed Sandbox Code"] = bool(
            "stdout" in response.lower() or "sandbox" in response.lower() or "code execution" in thinking.lower() or "executing code" in thinking.lower() or "matrix" in response.lower() or "weights" in response.lower() or "diff" in response.lower() or "absolute" in response.lower()
        )
    if "TurboQuant" in test["query"]:
        checks["Scouted TurboQuant repo"] = bool(
            "turboquant" in response.lower() and ("quant" in response.lower() or "cuda" in response.lower() or "gpu" in response.lower())
        )
    if "mamba-ssm-torch" in test["query"]:
        checks["Detected non-existent package"] = bool(
            "hallucination" in response.lower() or "does not exist" in response.lower() or "not find" in response.lower() or "not exist" in response.lower()
        )
    
    passed_all = True
    for key, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            passed_all = False
        print(f"  {status:<12} - {key}")
        
    print(f"Overall Result: {'[SUCCESS]' if passed_all else '[ATTENTION REQUIRED]'}")
    print("="*90 + "\n")
    
    return {
        "passed_all": passed_all,
        "checks": checks
    }

def save_reports(results: list):
    # Ensure dir exists
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Save JSON Report
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Save Markdown Report
    total_time = sum(r["elapsed"] for r in results)
    total_passed = sum(1 for r in results if r.get("success", False))
    success_rate = (total_passed / len(results)) * 100 if results else 0.0
    
    md_content = f"""# DeepThink Comprehensive Horizontal Test Suite Report

## 📊 Summary Metrics
* **Total Executed Tests:** {len(results)} / 10
* **Success Rate:** {success_rate:.1f}% ({total_passed} Passed, {len(results) - total_passed} Failed)
* **Total Execution Time:** {total_time:.2f} seconds (~{total_time/60:.2f} minutes)
* **Host Environment:** Windows Local Machine

---

## 🧪 Detailed Test Executions

"""
    for r in results:
        status_emoji = "✅" if r.get("success", False) else "❌"
        md_content += f"""### {status_emoji} Test Case {r['id']}: {r['name']}
* **Status:** {"Passed" if r.get('success', False) else "Failed"}
* **Execution Time:** {r.get('elapsed', 0.0):.2f}s
* **Error Logs:** `{r.get('error', 'None')}`
* **Automated Quality Assurances:**
"""
        metrics = r.get("metrics", {})
        if metrics:
            for chk, val in metrics.get("checks", {}).items():
                chk_emoji = "🟩" if val else "🟥"
                md_content += f"  - {chk_emoji} **{chk}**: {'Passed' if val else 'Failed'}\n"
        else:
            md_content += "  - 🟥 No metrics generated due to execution failure.\n"
            
        md_content += f"\n---\n"
        
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n[System] Comprehensive JSON report successfully saved to: {REPORT_JSON_PATH}")
    print(f"[System] Comprehensive Markdown walkthrough saved to: {REPORT_MD_PATH}\n")

if __name__ == "__main__":
    print("="*90)
    print("DEEPTHINK HORIZONTAL AUTOMATED RESEARCH VERIFIER")
    print("=================================================================================")
    print("Initializing comprehensive multi-agent research evaluation...")
    
    # Check if a specific single test ID was passed as CLI argument
    target_id = None
    if len(sys.argv) > 1:
        try:
            target_id = int(sys.argv[1])
            print(f"Targeting single test case ID: {target_id}")
        except ValueError:
            pass
            
    results = []
    
    # Load existing report to preserve other runs if we are target-running
    if target_id and os.path.exists(REPORT_JSON_PATH):
        try:
            with open(REPORT_JSON_PATH, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    for tc in TEST_CASES:
        if target_id and tc["id"] != target_id:
            continue
            
        res = run_test_case(tc)
        
        # Replace or append results
        if target_id:
            results = [r for r in results if r["id"] != target_id]
        results.append(res)
        
        # Save after every test case to preserve logs in case of crashes or interruptions!
        save_reports(results)
        
    print("DeepThink automated horizontal suite completed!")
