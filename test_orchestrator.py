import requests, time, json

url = "http://localhost:8000/v1/chat/completions"
start = time.time()
print("Sending request to orchestrator...")
r = requests.post(
    url,
    json={
        "model": "deepthink",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 100,
        "num_explorers": 2,
    },
    timeout=600,
)
elapsed = time.time() - start
print(f"Status: {r.status_code}, Time: {elapsed:.1f}s")
print(f"Response: {r.text[:1000]}")
