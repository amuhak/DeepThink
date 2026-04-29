import requests, json

url = "http://localhost:8000/v1/chat/completions"

# Test query for a deep technical dive
query = "Explain the PolarQuant step in the TurboQuant paper in detail, providing the mathematical formulas."

print(f"Testing elite technical research for query: {query}\n")

try:
    req = requests.post(
        url,
        json={
            "model": "deepthink",
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
            "stream": True,
            "num_explorers": 1, 
        },
        stream=True,
        timeout=600,
    )

    for line in req.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                try:
                    data = json.loads(decoded[6:])
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                except:
                    pass
except Exception as e:
    print(f"\nError: {e}")
