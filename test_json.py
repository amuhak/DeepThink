import requests, json

url = "https://llm.amuhak.com/v1/chat/completions"

r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-27B",
        "messages": [
            {
                "role": "system",
                "content": 'Return valid JSON only. No markdown fences. Structure: {"plan": "string", "prompts": [{"text": "string", "type": "prove"}]}',
            },
            {"role": "user", "content": "Problem: What is 2+2?"},
        ],
        "temperature": 0.3,
        "max_tokens": 49152,
        "response_format": {"type": "json_object"},
    },
    timeout=300,
)

data = r.json()
msg = data["choices"][0]["message"]
print(f"Finish: {data['choices'][0]['finish_reason']}")
print(f"Reasoning length: {len(msg.get('reasoning_content', ''))}")
print(f"Content length: {len(msg.get('content', ''))}")
print(f"--- Content ---")
print(msg.get("content", "")[:1000])
