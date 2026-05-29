import os
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN not found in .env")
    exit(1)

api = HfApi(token=HF_TOKEN)

# 1. Get user info to determine namespace
user_info = api.whoami()
namespace = user_info["name"]
repo_id = f"{namespace}/maged-ai-bot"

print(f"Creating Space: {repo_id}...")

# 2. Create the Space
try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=True,
        exist_ok=True
    )
    print("Space created or already exists.")
except Exception as e:
    print(f"Failed to create space: {e}")
    exit(1)

# 3. Add secrets
secrets_to_add = [
    "TELEGRAM_BOT_TOKEN", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
    "OLLAMA_API_KEYS", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
    "GEMINI_API_KEYS", "GOOGLE_SERVICE_ACCOUNT_JSON", "WHATSAPP_TOKEN",
    "WHATSAPP_PHONE_ID", "WHATSAPP_VERIFY_TOKEN", "META_APP_ID", "META_APP_SECRET"
]

print("\nUploading secrets...")
for secret in secrets_to_add:
    val = os.getenv(secret)
    if val:
        try:
            api.add_space_secret(repo_id=repo_id, key=secret, value=val)
            print(f"OK: {secret}")
        except Exception as e:
            print(f"FAIL: {secret} -> {e}")
    else:
        print(f"SKIP: {secret} (empty)")

# 4. Upload files
print("\nUploading files...")
files_to_upload = ["Dockerfile", "requirements.txt", "service-account.json"]
# Also upload api folder
api.upload_folder(
    folder_path="api",
    path_in_repo="api",
    repo_id=repo_id,
    repo_type="space"
)
print("Uploaded api/ folder.")

for f in files_to_upload:
    if os.path.exists(f):
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f,
            repo_id=repo_id,
            repo_type="space"
        )
        print(f"Uploaded {f}.")

print(f"\nAll done! Your space is building at: https://huggingface.co/spaces/{repo_id}")
print(f"Direct URL for webhooks will be: https://{namespace}-maged-ai-bot.hf.space")
