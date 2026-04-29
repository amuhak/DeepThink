import requests, time

url = "http://localhost:8001/v1/chat/completions"

# Simulate the evaluator prompt size
evaluator_prompt = """## Original Problem
How many distinct sequences of length 6 can a knight starting on 'Q' visit on a standard QWERTY keyboard?

## Current Plan
Fallback plan (JSON parse failed)

## Worker Outputs (last 1000 chars each)

### Worker 0 (prove)
FINAL ANSWER: 12345

### Worker 1 (prove)
FINAL ANSWER: 67890

### Worker 2 (refute)
FINAL ANSWER: Cannot determine

### Worker 3 (refute)
FINAL ANSWER: 0

## Execution Logs (last 5)

### Log 0 (Worker 0)
stdout: 12345
stderr:
Exit: 0

### Log 1 (Worker 1)
stdout: 67890
stderr:
Exit: 0
"""

start = time.time()
r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-27B",
        "messages": [
            {
                "role": "system",
                "content": """You are the Advisor - a ruthless evaluator of research results. Output JSON only:
{"status": "SOLVED|RETRY|PIVOT", "critique": "...", "final_answer": "answer or null"}""",
            },
            {"role": "user", "content": evaluator_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 49152,
        "response_format": {"type": "json_object"},
    },
    timeout=300,
)
elapsed = time.time() - start
print(f"Status: {r.status_code}, Time: {elapsed:.1f}s")
if r.status_code == 200:
    data = r.json()
    msg = data["choices"][0]["message"]
    print(f"Content: {msg.get('content', '')[:500]}")
else:
    print(r.text[:500])
