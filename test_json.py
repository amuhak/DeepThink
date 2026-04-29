import sys

sys.path.insert(0, "/app")
import asyncio
from llm_client import pro_client


async def test():
    print("Testing invoke_json...", flush=True)
    resp = await pro_client.invoke_json(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": 'Return a JSON with key "answer" and value "hello"',
            },
        ]
    )
    print("Response:", resp.content[:200])
    print("Timed out:", resp.timed_out)


asyncio.run(test())
