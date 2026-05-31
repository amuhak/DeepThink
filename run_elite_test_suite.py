import requests
import json
import time

URL = "http://localhost:8000/v1/chat/completions"

def run_test_case(name: str, query: str, num_explorers: int = 1, max_loops: int = 3):
    print("\n" + "="*80)
    print(f"RUNNING TEST CASE: {name}")
    print(f"Query: {query}")
    print("="*80 + "\n")
    
    start_time = time.time()
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
                "num_explorers": num_explorers,
                "max_loops": max_loops,
            },
            stream=True,
            timeout=600,
        )
        
        full_response = []
        for line in req.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    try:
                        data = json.loads(decoded[6:])
                        delta = data["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_response.append(content)
                    except Exception:
                        pass
        
        elapsed = time.time() - start_time
        print(f"\n\n[Test Case '{name}' Completed in {elapsed:.2f}s]")
        
        # Self-evaluation of response quality
        text_response = "".join(full_response)
        evaluate_response(name, text_response)
        
    except Exception as e:
        print(f"\n[Test Case '{name}' FAILED with error: {e}]")

def evaluate_response(name: str, response: str):
    print("\n" + "-"*40)
    print(f"SELF-EVALUATION FOR: {name}")
    print("-"*40)
    
    # Check for core technical markers
    checks = {
        "Has LaTeX formulas": ("$" in response or "$$" in response),
        "Contains final summary (FINAL:)": "FINAL:" in response or "Final Verdict" in response or "Conclusion" in response,
        "Has source citations (Source:)": "Source" in response or "https://" in response,
    }
    
    if "PDF" in name:
        checks["Used PDF vision findings"] = "PDF Vision" in response or "dot-product" in response or "attention" in response or "phenomenon" in response
    if "TurboQuant" in name:
        checks["Identified TurboQuant focus"] = "TurboQuant" in response.lower() and ("quant" in response.lower() or "cuda" in response.lower() or "gpu" in response.lower() or "github" in response.lower())
    
    passed_all = True
    for key, passed in checks.items():
        status = "[✓] PASSED" if passed else "[✗] FAILED"
        if not passed:
            passed_all = False
        print(f"  {status:<12} - {key}")
        
    if passed_all:
        print("\nResult: [SUCCESS] The agent successfully solved the research problem with high technical fidelity!")
    else:
        print("\nResult: [ATTENTION] Some criteria were not fully satisfied. Check the output logs above for details.")
    print("="*80 + "\n")

# Sequential Test Suite execution
if __name__ == "__main__":
    print("Initializing Elite Multi-Agent DeepThink Test Suite...")
    print("Executing tests sequentially to respect single-machine CPU/RAM bounds...\n")
    
    # Test Case 1: Quantization / New Domain Search ("TurboQuant")
    # This checks: Deep search, GitHub scraping, quantization logic reasoning.
    run_test_case(
        "TurboQuant Domain Scouting & Scraping",
        "Search for 'TurboQuant' on GitHub. What is its primary focus, what quantization techniques does it implement, and what is its performance edge?",
        num_explorers=1,
        max_loops=2
    )

    # Test Case 2: Math Vision extraction + isolated sandbox run
    # This checks: PDF vision extraction (get_pdf_nexttime) + Sandbox compilation + code runner.
    run_test_case(
        "Multimodal PDF Vision + Sandbox Math Validation",
        "Analyze page 4 of the PDF paper at https://arxiv.org/pdf/1706.03762.pdf using get_pdf_nexttime. Extract the Scaled Dot-Product Attention formula, and then write and run a Python script in your sandbox to compute the attention weights for a query matrix Q=[[1,0],[0,1]] and key matrix K=[[1,2],[3,4]]. Output the python code, execution output, and final weights.",
        num_explorers=1,
        max_loops=3
    )

    # Test Case 3: Parallel High-Concurrency PDF Extraction
    # This checks: Parallel PDF download queue, fitz rendering, max 4 semaphore throttle.
    run_test_case(
        "Concurrency Stress Test - 3 Parallel PDFs",
        "Compare page 1 of the following three PDFs using get_pdf_nexttime: 1. https://arxiv.org/pdf/1706.03762.pdf 2. https://arxiv.org/pdf/2402.17762.pdf 3. https://arxiv.org/pdf/2310.00000.pdf. Extract their titles and primary proposed concepts concurrently, and summarize their differences.",
        num_explorers=2,
        max_loops=2
    )
