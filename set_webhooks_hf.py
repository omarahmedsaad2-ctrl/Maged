import os
import requests
from dotenv import load_dotenv

load_dotenv()

# The permanent Hugging Face space URL
URL = "https://omasfomap-maged-ai-bot.hf.space"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
META_APP_ID = os.getenv("META_APP_ID")
META_APP_SECRET = os.getenv("META_APP_SECRET")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

print("1. Setting Telegram Webhook...")
r1 = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={URL}/webhook")
print(f"Telegram: {r1.status_code} - {r1.text}")

print("\n2. Setting WhatsApp Webhook...")
r2 = requests.post(
    f"https://graph.facebook.com/v25.0/{META_APP_ID}/subscriptions",
    data={
        "object": "whatsapp_business_account",
        "callback_url": f"{URL}/whatsapp-webhook",
        "verify_token": WHATSAPP_VERIFY_TOKEN,
        "fields": "messages",
        "access_token": f"{META_APP_ID}|{META_APP_SECRET}"
    }
)
print(f"WhatsApp: {r2.status_code} - {r2.text}")

print("\nAll webhooks updated to HF Space permanently!")
