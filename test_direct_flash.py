import requests, json

url = "http://localhost:8000/v1/chat/completions"

# Test direct access to 'flash' model (no orchestrator)
print("--- Testing Direct Flash Access ---")
req = requests.post(url, json={
    "model": "flash",
    "messages": [{"role": "user", "content": "Hello, who are you? Tell me in 10 words."}],
    "stream": True
}, stream=True)

for line in req.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith('data: '):
            try:
                data = json.loads(decoded[6:])
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
            except:
                pass
print("\n")
