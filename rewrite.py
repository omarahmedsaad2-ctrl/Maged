import re

with open(r"c:\Users\LILMAR\Desktop\Maged\ai-bot\api\index.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add httpx import and global client
if "import httpx" not in code:
    code = code.replace("import requests", "import requests\nimport httpx\nimport asyncio")

if "async_client =" not in code:
    code = code.replace("app = FastAPI()", "app = FastAPI()\n\n# Global async HTTP client with pooling\nasync_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=20), timeout=httpx.Timeout(120.0))")

# 2. Make functions async
code = code.replace("def get_embedding(", "async def get_embedding(")
code = code.replace("def get_query_embedding(", "async def get_query_embedding(")
code = code.replace("def search_similar(", "async def search_similar(")
code = code.replace("def run_sync_logic(", "async def run_sync_logic(")
code = code.replace("def ask_ollama(", "async def ask_ollama(")
code = code.replace("def send_telegram(", "async def send_telegram(")
code = code.replace("def send_telegram_typing(", "async def send_telegram_typing(")
code = code.replace("def send_telegram_keyboard(", "async def send_telegram_keyboard(")
code = code.replace("def mark_whatsapp_read(", "async def mark_whatsapp_read(")
code = code.replace("def send_whatsapp(", "async def send_whatsapp(")
code = code.replace("def get_rag_response(", "async def get_rag_response(")
code = code.replace("def process_telegram_message(", "async def process_telegram_message(")
code = code.replace("def process_whatsapp_message(", "async def process_whatsapp_message(")

# 3. Replace requests.post with await async_client.post
code = code.replace("requests.post(", "await async_client.post(")

# 4. Await function calls inside other functions
code = code.replace("emb = get_embedding(chunk)", "emb = await get_embedding(chunk)")
code = code.replace("embedding = get_query_embedding(query)", "embedding = await get_query_embedding(query)")
code = code.replace("search_similar(text", "await search_similar(text")
code = code.replace("answer = get_rag_response(", "answer = await get_rag_response(")
code = code.replace("send_telegram(chat_id", "await send_telegram(chat_id")
code = code.replace("send_telegram_typing(chat_id", "await send_telegram_typing(chat_id")
code = code.replace("send_telegram_keyboard(chat_id", "await send_telegram_keyboard(chat_id")
code = code.replace("send_whatsapp(phone", "await send_whatsapp(phone")
code = code.replace("mark_whatsapp_read(msg_id)", "await mark_whatsapp_read(msg_id)")
code = code.replace("run_sync_logic(chat_id)", "await run_sync_logic(chat_id)")
code = code.replace("report = run_sync_logic()", "report = await run_sync_logic()")

with open(r"c:\Users\LILMAR\Desktop\Maged\ai-bot\api\index_async.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Rewrote to async successfully!")
