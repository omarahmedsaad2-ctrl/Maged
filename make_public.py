import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
repo_id = "omasfomap/maged-ai-bot"

print(f"Making Space {repo_id} PUBLIC via REST API...")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

r = requests.put(
    f"https://huggingface.co/api/spaces/{repo_id}/settings",
    headers=headers,
    json={"private": False}
)

if r.status_code == 200:
    print("Space is now PUBLIC! 🌍")
else:
    print(f"Error: {r.status_code} - {r.text}")
