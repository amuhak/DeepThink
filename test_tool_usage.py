import requests, json

url = "http://localhost:8000/v1/chat/completions"

# Test query for a real tool call
query = "Search for 'TurboQuant' on GitHub and tell me what its primary focus is."

print(f"Testing tool usage for query: {query}\n")

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
            "num_explorers": 1,  # Minimal explorers for speed
        },
        stream=True,
        timeout=300,
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

                    # Stop early once we see a search being executed or results coming in
                    if (
                        "[Worker 0 searching:" in content
                        or "Search results for" in content
                    ):
                        print("\n\n[SUCCESS: Tool call detected!]")
                        req.close()
                        break
                except:
                    pass
except Exception as e:
    print(f"\nError: {e}")
