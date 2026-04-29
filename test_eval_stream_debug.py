import requests, json, re

url = "http://localhost:8000/v1/chat/completions"

req = requests.post(
    url,
    json={
        "model": "deepthink",
        "messages": [
            {"role": "user", "content": "What is the prime factorization of 100?"}
        ],
        "stream": True,
        "num_explorers": 2,
    },
    stream=True,
    timeout=1200,
)

content_parts = []
for line in req.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        content_parts.append(decoded)
        if decoded.startswith("data: "):
            json_str = decoded[6:]
            try:
                data = json.loads(json_str)
                if "error" in str(data):
                    print(f"Error: {data}")
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(f"Content chunk: {repr(content[:100])}")
                finish_reason = data["choices"][0].get("finish_reason")
                if finish_reason:
                    print(f"\n[Finished: {finish_reason}]")
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")
                print(f"Raw: {json_str[:300]}")
