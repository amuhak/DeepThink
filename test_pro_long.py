import requests, time

url = "https://llm.amuhak.com/v1/chat/completions"

PRO_PLANNER_SYSTEM = """You are the Advisor — a strategic research planner.

Your job is to take a complex problem and design a research plan using "Balanced Prompting".
You must generate exactly 4 sub-prompts for parallel worker models:
- Half should attempt to PROVE the premise
- Half should attempt to REFUTE or find counterexamples

Each prompt should instruct the worker to:
1. Think through the problem step by step
2. Write Python code to verify claims (use sympy for math, numerical methods for computation)
3. Use web search if background knowledge is needed
4. Return a final verdict with reasoning

Output must be valid JSON with this exact structure:
{
  "plan": "High-level research strategy",
  "prompts": [
    {"text": "Prompt for worker...", "type": "prove"},
    {"text": "Prompt for worker...", "type": "refute"}
  ]
}

Do NOT include markdown code fences around the JSON. Return raw JSON only."""

start = time.time()
r = requests.post(
    url,
    json={
        "model": "unsloth/Qwen3.6-27B",
        "messages": [
            {"role": "system", "content": PRO_PLANNER_SYSTEM},
            {
                "role": "user",
                "content": "Problem: How many distinct sequences of length 6 can a knight starting on 'Q' visit on a standard QWERTY keyboard?",
            },
        ],
        "temperature": 0.3,
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
