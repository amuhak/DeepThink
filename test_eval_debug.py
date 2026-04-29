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
            print(f"JSON: {json_str}")
            try:
                data = json.loads(json_str)
                if "error" in str(data):
                    print(f"Error: {data}")
                print(f"Top keys: {list(data.keys())}")
            except Exception as e:
                print(f"JSON parse error: {e}")
                print("Raw:", json_str[:300])
