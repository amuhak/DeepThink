import requests, time

url = "https://llm.prnt.ink/v1/chat/completions"

start = time.time()
r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-35B",
        "messages": [
            {
                "role": "system",
                "content": 'Output JSON only: {"status": "SOLVED|RETRY|PIVOT", "critique": "...", "final_answer": null}',
            },
            {
                "role": "user",
                "content": "Problem: What is 2+2? Worker 0 says 4. Worker 1 says 5. Evaluate.",
            },
        ],
        "temperature": 0.1,
        "max_tokens": 49152,
        "response_format": {"type": "json_object"},
    },
    timeout=300,
)
elapsed = time.time() - start
print(f"FLASH Status: {r.status_code}, Time: {elapsed:.1f}s")
if r.status_code == 200:
    data = r.json()
    msg = data["choices"][0]["message"]
    print(f"Content: {msg.get('content', '')[:300]}")
