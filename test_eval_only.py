import requests, json

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

for line in req.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data = json.loads(decoded[6:])
            print(f'Type: {data.get("type", "unknown")}')
            if "data" in data:
                print(f'Data keys: {data["data"].keys()}')
                print(
                    f'Delta content: {data["data"].get("delta", {}).get("content", "")[:200]}'
                )
            finish = data["data"].get("choices", [{}])[0].get("finish_reason")
            if finish:
                print(f"\n\n[Finished: {finish}]")
