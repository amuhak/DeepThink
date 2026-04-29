import asyncio
import httpx


async def test():
    async with httpx.AsyncClient(timeout=600) as client:
        # Use a prompt that requires current info - forces web search
        resp = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "deepthink",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the current version of Python as of 2026? Search the web to find the answer.",
                    }
                ],
                "stream": False,
                "num_explorers": 2,
                "max_loops": 3,
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Check if search results appear in the output
        has_search = (
            "python.org" in content.lower()
            or "Search results" in content
            or "Welcome to Python" in content
        )
        print("SEARCH USED:", has_search)
        print("Answer snippet:", content[:500])
        print("\nFull usage:", data.get("usage", {}))


asyncio.run(test())
