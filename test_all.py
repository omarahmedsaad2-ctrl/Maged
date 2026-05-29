import os
import sys
import json
from dotenv import load_dotenv

# Load env before importing index
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi.testclient import TestClient
from api.index import app, get_embedding, ask_ollama, get_drive_service, supabase, get_rag_response

client = TestClient(app)

def run_tests():
    print("--- Starting Full System Tests ---")

    # 1. Test Supabase Connection
    print("\n1. Testing Supabase...")
    try:
        res = supabase.table("whatsapp_chat_history").select("id").limit(1).execute()
        print("✅ Supabase connection successful.")
    except Exception as e:
        print(f"❌ Supabase failed: {e}")

    # 2. Test Gemini API (get_embedding)
    print("\n2. Testing Gemini Embeddings...")
    try:
        emb = get_embedding("Hello world")
        if isinstance(emb, list) and len(emb) == 768:
            print("✅ Gemini Embedding successful (768 dims).")
        else:
            print(f"❌ Gemini Embedding returned unexpected format: {type(emb)}")
    except Exception as e:
        print(f"❌ Gemini Embedding failed: {e}")

    # 3. Test Ollama API (ask_ollama)
    print("\n3. Testing Ollama API...")
    try:
        resp = ask_ollama("You are a helpful assistant.", "Reply with exactly the word 'OK'.")
        if resp and isinstance(resp, str):
            print(f"✅ Ollama API successful. Response snippet: {resp[:20]}")
        else:
            print("❌ Ollama API failed to return a valid string.")
    except Exception as e:
        print(f"❌ Ollama API failed: {e}")

    # 4. Test Google Drive Service
    print("\n4. Testing Google Drive Service Account...")
    try:
        service = get_drive_service()
        files = service.files().list(pageSize=1).execute()
        print("✅ Google Drive Auth successful.")
    except Exception as e:
        print(f"❌ Google Drive Auth failed: {e}")

    # 5. Test Telegram Webhook Endpoint
    print("\n5. Testing Telegram Webhook endpoint...")
    try:
        # Simulate a simple text message
        payload = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "from": {"id": 123, "is_bot": False, "first_name": "TestUser"},
                "chat": {"id": 123, "type": "private"},
                "date": 1600000000,
                "text": "/start"
            }
        }
        # Note: This will trigger send_telegram in the background. Since we don't want to actually spam a user,
        # we're using a fake chat ID. The API will just return 400 when it tries to send, which is handled gracefully.
        response = client.post("/webhook", json=payload)
        if response.status_code == 200:
            print("✅ Telegram Webhook processed successfully.")
        else:
            print(f"❌ Telegram Webhook returned {response.status_code}")
    except Exception as e:
        print(f"❌ Telegram Webhook test failed: {e}")

    # 6. Test WhatsApp Webhook Endpoint
    print("\n6. Testing WhatsApp Webhook endpoint...")
    try:
        # Simulate an incoming WA message
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"display_phone_number": "123", "phone_number_id": "123"},
                        "contacts": [{"profile": {"name": "Test User"}, "wa_id": "9999999999"}],
                        "messages": [{
                            "from": "9999999999",
                            "id": "wamid.test",
                            "timestamp": "1600000000",
                            "text": {"body": "test WA message"},
                            "type": "text"
                        }]
                    },
                    "field": "messages"
                }]
            }]
        }
        response = client.post("/whatsapp-webhook", json=payload)
        if response.status_code == 200:
            print("✅ WhatsApp Webhook processed successfully.")
        else:
            print(f"❌ WhatsApp Webhook returned {response.status_code}")
    except Exception as e:
        print(f"❌ WhatsApp Webhook test failed: {e}")

    print("\n--- All Tests Completed ---")

if __name__ == "__main__":
    run_tests()
