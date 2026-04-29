import requests

url = "https://llm.prnt.ink/v1/chat/completions"

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
                "content": "Problem: What is 2+2? Worker 0 says 4. Worker 1 says 5.",
            },
        ],
        "temperature": 0.1,
        "max_tokens": 49152,
        "response_format": {"type": "json_object"},
    },
    timeout=120,
)

data = r.json()
msg = data["choices"][0]["message"]
print(f"Reasoning length: {len(msg.get('reasoning_content', ''))}")
print(f"Content length: {len(msg.get('content', ''))}")
print(f"Content: [{msg.get('content', '')}]")
