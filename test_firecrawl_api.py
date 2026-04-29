import requests

url = "http://localhost:3002/v1/scrape"
payload = {"url": "https://example.com", "formats": ["markdown"]}

print(f"Testing Firecrawl API at {url}...")
try:
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Success! Data received:")
        print(resp.json())
    else:
        print(f"Failed: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
