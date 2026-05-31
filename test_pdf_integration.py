import requests, json

url = "http://localhost:8000/v1/chat/completions"

# Query that forces downloading the paper and using get_pdf_nexttime
query = "Analyze page 1 of the PDF at https://arxiv.org/pdf/2402.17762.pdf using get_pdf_nexttime to tell me the primary phenomenon it observes in 10 words."

print(f"Testing PDF integration flow for query: {query}\n")

try:
    req = requests.post(
        url,
        json={
            "model": "deepthink",
            "messages": [
                {
                    "role": "user",
                    "content": query,
                    "stream": True,
                }
            ],
            "stream": True,
            "num_explorers": 1,  # Speed up by using 1 explorer
            "max_loops": 2,
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
                except Exception as e:
                    pass
except Exception as e:
    print(f"\nError: {e}")
