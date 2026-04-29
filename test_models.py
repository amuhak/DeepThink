import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/v1/models")
        print("Models:", resp.json())

if __name__ == "__main__":
    asyncio.run(test())
