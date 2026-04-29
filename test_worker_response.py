import requests, time

url = "https://llm.prnt.ink/v1/chat/completions"

r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-35B",
        "messages": [
            {
                "role": "system",
                "content": "You are a research worker. Output code in ```python blocks and searches in ```search blocks. Your final answer must start with FINAL: on its own line.",
            },
            {
                "role": "user",
                "content": "What is the prime factorization of 100? Write Python code to verify.",
            },
        ],
        "temperature": 0.7,
        "max_tokens": 49152,
    },
    timeout=120,
)

data = r.json()
msg = data["choices"][0]["message"]
print(f"Finish: {data['choices'][0]['finish_reason']}")
print(f"Reasoning length: {len(msg.get('reasoning_content', ''))}")
print(f"Content length: {len(msg.get('content', ''))}")
print(f"\n--- Content ---")
print(msg.get("content", "")[:2000])
print(f"\n--- Reasoning (last 500 chars) ---")
print(msg.get("reasoning_content", "")[-500:])
