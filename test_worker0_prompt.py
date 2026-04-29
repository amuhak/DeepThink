import requests, json

url = "https://llm.prnt.ink/v1/chat/completions"

prompt = """Task: Calculate the number of distinct sequences of length 6 (visiting 6 keys) starting at 'Q' on a standard US QWERTY keyboard (alphanumeric block only).

1. Think step by step
2. Write Python code to verify claims
3. Use web search if needed
4. Return final verdict with reasoning

Your final answer must start with "FINAL:" on its own line."""

r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-35B",
        "messages": [
            {
                "role": "system",
                "content": "You are a research worker. Output code in ```python blocks.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 49152,
    },
    timeout=120,
)

data = r.json()
msg = data["choices"][0]["message"]
print(f"Status: {r.status_code}")
print(f"Finish: {data['choices'][0]['finish_reason']}")
print(f"Reasoning len: {len(msg.get('reasoning_content', ''))}")
print(f"Content len: {len(msg.get('content', ''))}")
print(f"\n--- Content (first 500 chars) ---")
print(msg.get("content", "")[:500])
