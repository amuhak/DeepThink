import requests, time

url = "https://llm.amuhak.com/v1/chat/completions"
start = time.time()
r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-27B",
        "messages": [{"role": "user", "content": "Say hello in 5 words"}],
        "max_tokens": 4096,
    },
    timeout=90,
)
elapsed = time.time() - start
data = r.json()
choice = data["choices"][0]
msg = choice["message"]
print(f"Status: {r.status_code}, Time: {elapsed:.1f}s")
print(f"Finish: {choice['finish_reason']}")
print(f"Content length: {len(msg.get('content', ''))}")
print(f"Reasoning length: {len(msg.get('reasoning_content', ''))}")
print(f"Content: {msg.get('content', '')[:300]}")
