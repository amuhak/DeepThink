import asyncio
import httpx
import time
import json
import os
import sys

BASE_URL = "http://localhost:8000/v1"
COMPLETIONS_URL = f"{BASE_URL}/chat/completions"

async def test_models_list():
    print("\n--- Testing Model Registry Listing ---")
    async with httpx.AsyncClient(timeout=10.0) as client:
        t0 = time.time()
        resp = await client.get(f"{BASE_URL}/models")
        dt = time.time() - t0
        
        if resp.status_code != 200:
            print(f"[FAIL] /v1/models returned status code {resp.status_code}")
            return False
            
        data = resp.json()
        print(f"Response time: {dt:.3f}s")
        print("Models list:", json.dumps(data, indent=2))
        
        model_ids = [m["id"] for m in data.get("data", [])]
        if "think" in model_ids and "deepthink" in model_ids:
            print("[PASS] Both 'think' and 'deepthink' models are registered correctly.")
            return True
        else:
            print(f"[FAIL] Active models list is incorrect: {model_ids}")
            return False

async def run_streaming_profile(name: str, prompt: str, expected_keywords: list = None) -> dict:
    print(f"\n--- Testing Profile: {name} (Streaming Mode) ---")
    print(f"Prompt: {prompt}")
    
    payload = {
        "model": "think",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    events = []
    reasoning_tokens = []
    content_tokens = []
    
    ttft = None
    usage_stats = None
    t0 = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", COMPLETIONS_URL, json=payload) as response:
                if response.status_code != 200:
                    print(f"[FAIL] HTTP Status Code: {response.status_code}")
                    return {"success": False, "error": f"Status code {response.status_code}"}
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    decoded = line.strip()
                    if decoded.startswith("data: "):
                        raw_data = decoded[6:]
                        if raw_data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(raw_data)
                        except Exception as e:
                            print(f"[WARN] Malformed JSON in SSE chunk: {raw_data} ({e})")
                            continue
                            
                        if "usage" in chunk and chunk["usage"]:
                            usage_stats = chunk["usage"]
                            
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                            
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = choices[0].get("finish_reason", None)
                        
                        if ttft is None and content:
                            ttft = time.time() - t0
                            print(f"Time to First Token (TTFT): {ttft:.3f}s")
                            
                        if content:
                            events.append(content)
                            print(content, end="", flush=True)
                            
                        if finish_reason:
                            print(f"\n[Finished: {finish_reason}]")
                            
    except Exception as e:
        print(f"\n[FAIL] Streaming exception: {e}")
        return {"success": False, "error": str(e)}
        
    total_time = time.time() - t0
    print(f"\nTotal elapsed time: {total_time:.3f}s")
    
    full_response = "".join(events)
    
    # Verify stream structure guidelines
    has_thinking_start = "<thinking>" in full_response
    has_thinking_end = "</thinking>" in full_response
    
    # Extract thinking part and content part
    thinking_part = ""
    content_part = ""
    if has_thinking_start and has_thinking_end:
        parts = full_response.split("</thinking>")
        thinking_part = parts[0].replace("<thinking>", "")
        content_part = parts[1]
    else:
        content_part = full_response
        
    # Validation checks
    keyword_matches = []
    if expected_keywords:
        for kw in expected_keywords:
            match = (
                kw.lower() in content_part.lower() 
                or kw.lower() in thinking_part.lower()
                or kw.lower() in content_part.replace(",", "").lower()
                or kw.lower() in thinking_part.replace(",", "").lower()
            )
            keyword_matches.append((kw, match))
            
    print(f"Validation: Has <thinking>: {has_thinking_start}")
    print(f"Validation: Has </thinking>: {has_thinking_end}")
    
    tool_logs = []
    for line in full_response.split("\n"):
        if "- [Think]" in line:
            tool_logs.append(line)
            print(f"Captured execution event: {line}")
            
    has_usage = usage_stats is not None and usage_stats.get("total_tokens", 0) > 0
    print(f"Validation: Has valid token usage stats: {has_usage} ({usage_stats})")
    
    success = has_thinking_start and has_thinking_end and has_usage
    if expected_keywords:
        all_kw_matched = all(match for _, match in keyword_matches)
        success = success and all_kw_matched
        print(f"Keyword matches: {keyword_matches}")
        
    return {
        "success": success,
        "name": name,
        "mode": "streaming",
        "ttft": ttft,
        "total_time": total_time,
        "has_thinking_start": has_thinking_start,
        "has_thinking_end": has_thinking_end,
        "thinking_len": len(thinking_part),
        "content_len": len(content_part),
        "tool_logs": tool_logs,
        "response_preview": content_part[:300] + ("..." if len(content_part) > 300 else ""),
        "keyword_matches": keyword_matches,
        "usage": usage_stats
    }

async def run_blocking_profile(name: str, prompt: str, expected_keywords: list = None) -> dict:
    print(f"\n--- Testing Profile: {name} (Blocking Mode) ---")
    print(f"Prompt: {prompt}")
    
    payload = {
        "model": "think",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(COMPLETIONS_URL, json=payload)
            if resp.status_code != 200:
                print(f"[FAIL] HTTP Status Code: {resp.status_code}")
                return {"success": False, "error": f"Status code {resp.status_code}"}
                
            data = resp.json()
            
    except Exception as e:
        print(f"[FAIL] Blocking request exception: {e}")
        return {"success": False, "error": str(e)}
        
    total_time = time.time() - t0
    print(f"Total elapsed time: {total_time:.3f}s")
    
    choices = data.get("choices", [])
    if not choices:
        print("[FAIL] Response choices array is empty.")
        return {"success": False, "error": "Empty choices"}
        
    message = choices[0].get("message", {})
    content = message.get("content", "")
    print(f"Response Content:\n{content}")
    
    # Validation checks
    keyword_matches = []
    if expected_keywords:
        for kw in expected_keywords:
            match = kw.lower() in content.lower() or kw.lower() in content.replace(",", "").lower()
            keyword_matches.append((kw, match))
            
    usage_stats = data.get("usage", {})
    has_usage = usage_stats is not None and usage_stats.get("total_tokens", 0) > 0
    print(f"Validation: Has valid token usage stats: {has_usage} ({usage_stats})")
    
    success = len(content) > 0 and has_usage
    if expected_keywords:
        all_kw_matched = all(match for _, match in keyword_matches)
        success = success and all_kw_matched
        print(f"Keyword matches: {keyword_matches}")
        
    return {
        "success": success,
        "name": name,
        "mode": "blocking",
        "ttft": None,
        "total_time": total_time,
        "content_len": len(content),
        "response_preview": content[:300] + ("..." if len(content) > 300 else ""),
        "keyword_matches": keyword_matches,
        "usage": usage_stats
    }

async def main():
    print("====================================================")
    print("    DEEPTHINK 'THINK' AGENT COMPREHENSIVE SUITE     ")
    print("====================================================")
    
    # 1. Model List Check
    registry_ok = await test_models_list()
    
    results = []
    
    # Profile A: Quick Sandbox Calculation (Factorial mod)
    # Expected result in final answer or code execution: 948537388
    res_a_stream = await run_streaming_profile(
        "Profile A (Sandbox - streaming)", 
        "Calculate 52! mod 1000000007. Write python code to solve it and present the final modulo output.", 
        ["948537388"]
    )
    results.append(res_a_stream)
    
    res_a_block = await run_blocking_profile(
        "Profile A (Sandbox - blocking)", 
        "Calculate 52! mod 1000000007. Solve it using Python.", 
        ["948537388"]
    )
    results.append(res_a_block)
    
    # Profile B: Simple Factual Search (e.g. Monaco Grand Prix winner 2024 or 2025 or general winner)
    res_b_stream = await run_streaming_profile(
        "Profile B (Search - streaming)", 
        "Who won the 2024 Monaco Grand Prix in Formula 1?", 
        ["Leclerc", "Monaco"]
    )
    results.append(res_b_stream)
    
    # Profile C: URL Scrape verification
    res_c_stream = await run_streaming_profile(
        "Profile C (Scrape - streaming)", 
        "Please fetch and scrape the URL 'https://raw.githubusercontent.com/prometheus/prometheus/main/LICENSE' and summarize the first sentence of the license.", 
        ["Apache", "License"]
    )
    results.append(res_c_stream)
    
    # Summarize and write comprehensive JSON output
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registry_ok": registry_ok,
        "results": results
    }
    
    # Write JSON report
    report_json_path = r"C:\Users\amuhak\.gemini\antigravity-cli\brain\4ea459c4-4af4-4f96-9fd7-deaed05aaf50\comprehensive_test_report.json"
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n====================================================")
    print("                 TEST SUITE SUMMARY                 ")
    print("====================================================")
    print(f"Registry Status: {'[PASS]' if registry_ok else '[FAIL]'}")
    
    passed_tests = 0
    total_tests = len(results)
    for r in results:
        status_str = "[PASS]" if r.get("success") else "[FAIL]"
        time_str = f"{r.get('total_time'):.2f}s" if r.get('total_time') else "N/A"
        print(f"- {r.get('name')}: {status_str} in {time_str}")
        if r.get("success"):
            passed_tests += 1
            
    print(f"\nFinal Score: {passed_tests} / {total_tests} passed.")
    print(f"Detailed JSON results written to: {report_json_path}")
    print("====================================================")

if __name__ == "__main__":
    asyncio.run(main())
