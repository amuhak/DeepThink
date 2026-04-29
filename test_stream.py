import requests, json

url = "http://localhost:8000/v1/chat/completions"

req = requests.post(
    url,
    json={
        "model": "deepthink",
        "messages": [
            {
                "role": "user",
                "content": "How many distinct sequences of length 6 can a knight starting on 'Q' visit on a standard QWERTY keyboard? Model the keyboard layout as a graph and count valid knight moves.",
            }
        ],
        "stream": True,
        "num_explorers": 4,
    },
    stream=True,
    timeout=1200,
)

for line in req.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data = json.loads(decoded[6:])
            delta = data["choices"][0]["delta"]
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
            finish = data["choices"][0].get("finish_reason")
            if finish:
                print(f"\n\n[Finished: {finish}]")
