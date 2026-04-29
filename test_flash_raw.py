import requests

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

print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type', 'unknown')}")
print(f"\n--- Raw response (first 500 chars) ---")
print(r.text[:500])
