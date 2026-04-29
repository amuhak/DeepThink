import requests, time

url = "https://llm.prnt.ink/v1/chat/completions"
start = time.time()
r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-35B",
        "messages": [
            {"role": "user", "content": "Calculate 2+2. Return only the number."}
        ],
        "max_tokens": 49152,
    },
    timeout=300,
)
elapsed = time.time() - start
data = r.json()
choice = data["choices"][0]
msg = choice["message"]
print(f"Status: {r.status_code}, Time: {elapsed:.1f}s")
print(f"Finish: {choice['finish_reason']}")
print(f"Content length: {len(msg.get('content', ''))}")
print(f"Reasoning length: {len(msg.get('reasoning_content', ''))}")
print(f"Content: {msg.get('content', '')[:500]}")
