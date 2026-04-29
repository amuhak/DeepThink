import requests

url = "https://llm.amuhak.com/v1/chat/completions"

r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-27B",
        "messages": [
            {
                "role": "system",
                "content": 'Output JSON only: {"plan": "string", "prompts": [{"text": "string", "type": "prove"}]}',
            },
            {"role": "user", "content": "Problem: What is 2+2?"},
        ],
        "temperature": 0.3,
        "max_tokens": 49152,
        "response_format": {"type": "json_object"},
    },
    timeout=120,
)

data = r.json()
msg = data["choices"][0]["message"]
print(f"Status: {r.status_code}")
print(f"Finish: {data['choices'][0]['finish_reason']}")
print(f"Content: {msg.get('content', '')[:500]}")
