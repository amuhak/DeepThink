import httpx
import asyncio
import os

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://code-sandbox:8000")


async def test():
    print("Testing Code Sandbox...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_URL}/execute",
                json={"code": "print('Hello from the Sandbox!')", "timeout": 5},
            )
            print(f"Sandbox response ({resp.status_code}):", resp.json())
    except Exception as e:
        print("Sandbox failed:", e)

    print("\nTesting SearXNG...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": "OpenAI", "format": "json"},
            )
            data = resp.json()
            results = data.get("results", [])
            print(
                f"SearXNG response ({resp.status_code}): found {len(results)} results"
            )
            if results:
                print(
                    "First result:", results[0].get("title"), "-", results[0].get("url")
                )
    except Exception as e:
        print("SearXNG failed:", e)


if __name__ == "__main__":
    asyncio.run(test())
