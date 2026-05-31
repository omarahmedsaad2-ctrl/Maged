import requests

keys = [
    "b703f9e43cf34a468212c3d34ebe1a14.OtPIW2pTsw50a4yGBFdqUx0m",
    "a1794391da8b4a80843e4651f051fa58.EpWk_mEFwAzbcN-crra8IjS9"
]

for idx, key in enumerate(keys):
    print(f"\n--- Testing Key {idx} ---")
    try:
        response = requests.post(
            "https://ollama.com/api/chat",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-oss:120b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False
            },
            timeout=30
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Exception: {e}")
