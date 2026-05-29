"""
Telegram & WhatsApp RAG Bot - Ollama Cloud + Vector Search.
Fully async architecture using httpx.AsyncClient for reliable
outbound HTTP on Hugging Face Spaces.
"""
import os
import io
import re
import json
import time
import sys
import asyncio
import httpx
import requests  # kept only for supabase internal use
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from supabase import create_client, Client
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- Global async HTTP client (created at startup) ---
_client: httpx.AsyncClient = None


@app.on_event("startup")
async def startup_event():
    global _client
    transport = httpx.AsyncHTTPTransport(retries=3)
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=15.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=True,
        verify=False,
        transport=transport,
    )
    # Start WhatsApp queue worker (processes messages sequentially)
    asyncio.create_task(_wa_queue_worker())
    print("===== Async HTTP client + WA queue worker ready =====", flush=True)


# --- WhatsApp Queue (sequential processing like Telegram inline) ---
_wa_queue = asyncio.Queue()


async def _wa_queue_worker():
    """Process WhatsApp messages one at a time from the queue."""
    print("[WA Worker] Started, waiting for messages...", flush=True)
    while True:
        try:
            data = await _wa_queue.get()
            await process_whatsapp_message(data)
        except Exception as e:
            print(f"[WA Worker] Error: {e}", flush=True)
        finally:
            _wa_queue.task_done()


@app.on_event("shutdown")
async def shutdown_event():
    global _client
    if _client:
        await _client.aclose()


# --- Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "maged_bot_secure_token")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OLLAMA_KEYS_RAW = os.getenv("OLLAMA_API_KEYS", os.getenv("OLLAMA_API_KEY", ""))
OLLAMA_KEYS = [k.strip().replace('\ufeff', '').replace('\r', '').replace('\n', '') for k in OLLAMA_KEYS_RAW.split(",") if k.strip()]
current_ollama_key_index = 0

GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]
current_gemini_key_index = 0
FOLDER_ID = "1xBpOgVa6gDT2MyZngiygXTYODC3LfvUu"
ADMIN_CHAT_IDS = [8284113566, 5103350500]

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- Helper: run sync supabase/drive calls in thread ---
async def db(fn):
    """Run a synchronous callable in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(fn)


# --- Embedding via Gemini API ---
async def get_embedding(text):
    """Get 768-dim embedding using Gemini API with failover."""
    global current_gemini_key_index
    if not GEMINI_KEYS:
        raise ValueError("No Gemini API keys found in .env (GEMINI_API_KEYS)")

    for attempt in range(len(GEMINI_KEYS)):
        current_key = GEMINI_KEYS[current_gemini_key_index]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={current_key}"

        try:
            resp = await _client.post(url, json={
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [{"text": text}]
                },
                "outputDimensionality": 768
            })

            if resp.status_code == 200:
                data = resp.json()
                return data["embedding"]["values"]
            else:
                print(f"Gemini key {current_gemini_key_index} failed with {resp.status_code}: {resp.text}", flush=True)
                current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_KEYS)
        except Exception as e:
            print(f"Gemini key {current_gemini_key_index} network error: {e}", flush=True)
            current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_KEYS)

    raise Exception("All Gemini API keys failed.")


async def get_query_embedding(text):
    """Get embedding for a query."""
    return await get_embedding(text)


# --- Vector Search in Supabase ---
async def search_similar(query, limit=5):
    """Search for similar documents using vector similarity."""
    embedding = await get_query_embedding(query)

    result = await db(lambda: supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_threshold": 0.3,
        "match_count": limit
    }).execute())

    return result.data if result.data else []


# --- Sync Logic (for Cron Job) ---
def get_drive_service():
    service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    info = None
    if service_account_info:
        try:
            info = json.loads(service_account_info)
        except Exception as e:
            print("Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON:", e, flush=True)

    if info:
        creds = service_account.Credentials.from_service_account_info(info)
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "service-account.json")
        creds = service_account.Credentials.from_service_account_file(path)

    return build('drive', 'v3', credentials=creds.with_scopes(
        ['https://www.googleapis.com/auth/drive.readonly']))


async def run_sync_logic(chat_id=None):
    if chat_id:
        await send_telegram(chat_id, "بدأ عملية المزامنة (Restore)... جاري فحص الملفات.")

    service = get_drive_service()
    results = await asyncio.to_thread(lambda: service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute())
    files = results.get('files', [])

    if not files:
        if chat_id:
            await send_telegram(chat_id, "لم يتم العثور على أي ملفات PDF.")
        return "No PDF files found."

    # Fetch already synced files
    synced_response = await db(lambda: supabase.table("synced_files").select("file_id").execute())
    synced_file_ids = {row["file_id"] for row in synced_response.data}

    files_to_sync = [f for f in files if f['id'] not in synced_file_ids]

    if not files_to_sync:
        if chat_id:
            await send_telegram(chat_id, "كل الملفات تم عمل sync لها مسبقاً. مفيش ملفات جديدة.")
        return "All files already synced."

    if chat_id:
        await send_telegram(chat_id, f"تم إيجاد {len(files_to_sync)} ملف جديد. جاري التحويل...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    total_chunks = 0

    for file_info in files_to_sync:
        if chat_id:
            await send_telegram(chat_id, f"جاري معالجة: {file_info['name']}")

        request_obj = service.files().get_media(fileId=file_info['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_obj)

        def _download():
            nonlocal fh, downloader
            done = False
            while not done:
                _, done = downloader.next_chunk()
        await asyncio.to_thread(_download)

        fh.seek(0)
        reader = PdfReader(fh)
        text = "".join([p.extract_text() or "" for p in reader.pages])

        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            emb = await get_embedding(chunk)
            await db(lambda c=chunk, e=emb, fn=file_info['name'], idx=i: supabase.table("documents").insert({
                "content": c,
                "embedding": e,
                "metadata": json.dumps({"source": fn, "chunk": idx})
            }).execute())
            total_chunks += 1

        # Mark as synced
        await db(lambda fid=file_info['id'], fname=file_info['name']: supabase.table("synced_files").insert({
            "file_id": fid,
            "file_name": fname
        }).execute())

    if chat_id:
        await send_telegram(chat_id, f"تم الانتهاء بنجاح! تمت معالجة {len(files_to_sync)} ملف ({total_chunks} أجزاء).")

    return f"Synced {len(files_to_sync)} files ({total_chunks} chunks)."


# --- Send Functions (all async with retry + hard timeouts) ---
async def _tg_post(url, json_data, timeout=30):
    """POST to Telegram API with a hard asyncio timeout."""
    return await asyncio.wait_for(_client.post(url, json=json_data), timeout=timeout)


async def send_telegram(chat_id, text):
    for i in range(0, len(str(text)), 4000):
        chunk = str(text)[i:i+4000]
        for attempt in range(3):
            try:
                r = await _tg_post(f"{TELEGRAM_API}/sendMessage",
                    {"chat_id": chat_id, "text": chunk}, timeout=30)
                if r.status_code == 200:
                    print(f"[TG SEND OK] to {chat_id}", flush=True)
                    break
                print(f"[TG SEND ERROR] attempt {attempt+1}: {r.status_code}: {r.text}", flush=True)
            except asyncio.TimeoutError:
                print(f"[TG SEND TIMEOUT] attempt {attempt+1}: hard 30s timeout for {chat_id}", flush=True)
            except Exception as e:
                print(f"[TG SEND FAIL] attempt {attempt+1}: {type(e).__name__}: {repr(e)}", flush=True)
            if attempt < 2:
                await asyncio.sleep(1)


async def send_telegram_typing(chat_id):
    try:
        await _tg_post(f"{TELEGRAM_API}/sendChatAction",
            {"chat_id": chat_id, "action": "typing"}, timeout=5)
        print(f"[TG TYPING OK] {chat_id}", flush=True)
    except asyncio.TimeoutError:
        print(f"[TG TYPING TIMEOUT] 5s hard timeout for {chat_id}", flush=True)
    except Exception as e:
        print(f"[TG TYPING FAIL] {type(e).__name__}: {repr(e)}", flush=True)


async def send_telegram_keyboard(chat_id, text, keyboard):
    for attempt in range(3):
        try:
            r = await _tg_post(f"{TELEGRAM_API}/sendMessage",
                {"chat_id": chat_id, "text": text, "reply_markup": keyboard}, timeout=30)
            if r.status_code == 200:
                break
            print(f"[TG KEYBOARD ERROR] attempt {attempt+1}: {r.status_code}: {r.text}", flush=True)
        except asyncio.TimeoutError:
            print(f"[TG KEYBOARD TIMEOUT] attempt {attempt+1}: 30s timeout for {chat_id}", flush=True)
        except Exception as e:
            print(f"[TG KEYBOARD FAIL] attempt {attempt+1}: {type(e).__name__}: {repr(e)}", flush=True)
        if attempt < 2:
            await asyncio.sleep(1)


async def mark_whatsapp_read(message_id):
    """Mark message as read (blue ticks). Uses fresh connection."""
    if not WHATSAPP_PHONE_ID or not WHATSAPP_TOKEN:
        return
    try:
        url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_ID}/messages"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=5, verify=False) as c:
            await c.post(url, headers=headers, json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            })
    except Exception:
        pass  # Blue ticks are non-critical


async def send_whatsapp(to_phone, text):
    """Send WhatsApp message using a fresh connection per attempt (avoids HF stale pool)."""
    if not WHATSAPP_PHONE_ID or not WHATSAPP_TOKEN:
        print("WhatsApp credentials missing.", flush=True)
        return
    url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    for i in range(0, len(str(text)), 4000):
        chunk = str(text)[i:i+4000]
        data = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": chunk}
        }
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30, verify=False) as c:
                    r = await c.post(url, headers=headers, json=data)
                if r.status_code == 200:
                    print(f"[WA SEND] OK to {to_phone}", flush=True)
                    break
                print(f"[WA SEND ERROR] attempt {attempt+1}: {r.status_code}: {r.text[:200]}", flush=True)
            except Exception as e:
                print(f"[WA SEND FAIL] attempt {attempt+1}: {type(e).__name__}: {repr(e)}", flush=True)
            if attempt < 2:
                await asyncio.sleep(2)


# --- RAG Response ---
GREETINGS = {"hi", "hello", "hey", "هلا", "اهلا", "مرحبا", "هاي", "السلام عليكم", "ازيك", "ازيكم", "صباح الخير", "مساء الخير", "يا مستر", "مستر"}


async def get_rag_response(user_id, text, history_table, user_column):
    # Smart: skip RAG search for greetings/short messages
    text_lower = text.strip().lower()
    is_greeting = text_lower in GREETINGS or len(text_lower) < 4

    if is_greeting:
        similar_docs = []
        print(f"[SMART] Skipped RAG for greeting: '{text_lower}'", flush=True)
    else:
        # Use fewer docs for short questions, more for complex ones
        doc_limit = 5 if len(text) < 30 else 10
        similar_docs = await search_similar(text, limit=doc_limit)
        print(f"[SMART] RAG search with {doc_limit} docs for: '{text[:40]}'", flush=True)

    def get_source(d):
        meta = d.get('metadata', {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except:
                meta = {}
        return meta.get('source', 'Unknown') if isinstance(meta, dict) else 'Unknown'

    available_sources = set()
    if similar_docs:
        for d in similar_docs:
            available_sources.add(get_source(d))
        context = "\n\n".join([f"[{get_source(d)}]\n{d['content']}" for d in similar_docs])
    else:
        result = await db(lambda: supabase.table("documents").select("content, metadata").limit(20).execute())
        context = "\n\n".join([f"[{get_source(d)}]\n{d['content']}" for d in result.data])[:8000]
        for d in result.data:
            available_sources.add(get_source(d))

    sources_list = ", ".join(sorted(available_sources)) if available_sources else "المنهج"

    opus_prompt_path = os.path.join(os.path.dirname(__file__), "OPUS_Universal_System_Prompt.md")
    opus_content = ""
    try:
        with open(opus_prompt_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            if "```text" in raw_content:
                opus_content = raw_content.split("```text")[1].split("```")[0].strip()
            else:
                opus_content = raw_content.strip()
    except Exception as e:
        opus_content = "You are OPUS. Act as a professional assistant."

    system_prompt = f"""{opus_content}

---

## YOUR ROLE: MR MAGED'S SMART ENGLISH TUTOR

IDENTITY:
- You are "مساعد مستر ماجد الذكي" — a friendly, knowledgeable English tutor for MR Maged's students.
- You cover all uploaded course materials: {sources_list}
- You know exactly what content you have. When asked, list available topics briefly.

LANGUAGE RULES (HIGHEST PRIORITY):
- If the student writes in Arabic → reply using a MIX where at least 60% of the message is in English. Use Egyptian Arabic only as short connectors and encouragements between English content. You are an English immersion tutor — push the student to absorb English naturally. Examples:
  "بص يا بطل، the present simple is used for habits and daily routines, يعني things you do regularly like: I wake up at 7, I go to school every day. Notice how the verb stays in its base form ركز في النقطة دي كويس"
  "الكلمة دي means 'opportunity' — it's when you have a chance to do something. For example: This is a great opportunity to improve your English. فاهم يا بطل؟"
  "Let me explain it simply كده... When you want to talk about something happening right now, you use the present continuous: subject + am/is/are + verb-ing. مثال: I am studying English right now"
- If the student writes in English → reply ENTIRELY in English at a B2 to C2+ level. Use rich vocabulary, varied sentence structures, idiomatic expressions, and natural academic English. Challenge the student to level up.
- Grammar terms, vocabulary, definitions, and examples MUST always be in English regardless of reply language.

MEMORY:
- You HAVE full conversation memory. The previous messages are REAL past messages with this specific student. 
- You MUST remember what was discussed. If the student says you talked before, ACKNOWLEDGE it. NEVER say you cannot remember.

PERSONALITY:
- Act exactly like MR Maged: relaxed, friendly, encouraging Egyptian English teacher.
- Give the core concept simply and directly. No textbook essays.
- Explain step by step using MR Maged's style and his exact words from the materials.
- GENDER AWARENESS: Detect the student's gender from their name or how they talk. Use the correct masculine or feminine forms in Arabic:
  📌 For boys: "بص يا بطل", "ركز معايا يا كبير", "يا بطبوط", "يا معلم", "برافو عليك"
  📌 For girls: "بصي يا بطلة", "ركزي معايا يا قمر", "يا كتكوتة", "يا ستي", "برافو عليكي"
  📌 Verb forms: use masculine (بتستخدم، عايز، فاهم) for boys, feminine (بتستخدمي، عايزة، فاهمة) for girls.
  📌 If unsure about gender, use gender-neutral terms like "يا فندم" or "يا صديقي".

RESPONSE FORMAT (CRITICAL - you are on Telegram/WhatsApp):
- Keep answers SHORT and conversational. Max 2-3 examples at a time.
- Use short lines. Each line should be roughly the same length.
- Use simple bullet points with emoji (📌, ✅, 🔹, 💡) instead of dashes or numbers.
- Separate sections with a blank line for readability.
- NEVER use markdown bold (asterisks **), tables, or headers (#).
- Keep text completely plain and clean.
- End with a friendly follow-up question when appropriate.

CONTENT RULES:
- Answer ANY English-related question, even outside MR Maged's files.
- PREFER examples from MR Maged's materials when available.
- Include Arabic translation for vocabulary if MR Maged provided one.
- NEVER mention source file names or document names unless asked.
- Minimize the word "unit" — don't mention it unless the student asks.
- If asked about non-English topics (coding, math, etc.) → politely decline:
  "بعتذر يا بطل، أنا هنا عشان أساعدك في الإنجليزي بس 😊 أقدر أساعدك في إيه في المنهج؟"

COURSE MATERIALS:
{context}
"""

    # Smart history: fewer messages for greetings, more for real questions
    history_limit = 6 if is_greeting else 15
    history = []
    try:
        history_response = await db(lambda: supabase.table(history_table).select("role, content").eq(user_column, user_id).order("created_at", desc=True).limit(history_limit).execute())
        history = list(reversed(history_response.data))
        print(f"[HISTORY] Loaded {len(history)}/{history_limit} messages for {user_id}", flush=True)
    except Exception as e:
        print("Failed to fetch chat history:", e, flush=True)

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": text})

    try:
        await db(lambda: supabase.table(history_table).insert({
            user_column: user_id,
            "role": "user",
            "content": text
        }).execute())
    except Exception as e:
        print("Failed to save user history:", e, flush=True)

    global current_ollama_key_index
    answer = "Sorry, I couldn't generate an answer."

    for attempt in range(max(1, len(OLLAMA_KEYS))):
        current_key = OLLAMA_KEYS[current_ollama_key_index] if OLLAMA_KEYS else ""
        try:
            response = await _client.post(
                "https://ollama.com/api/chat",
                json={
                    "model": "gpt-oss:120b",
                    "messages": messages,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {current_key}"},
            )
            if response.status_code == 200:
                answer = response.json().get("message", {}).get("content", "Sorry, I couldn't generate an answer.")
                break
            else:
                print(f"Ollama key {current_ollama_key_index} returned {response.status_code}, switching...", flush=True)
                if OLLAMA_KEYS:
                    current_ollama_key_index = (current_ollama_key_index + 1) % len(OLLAMA_KEYS)
                continue
        except Exception as e:
            print(f"Ollama key {current_ollama_key_index} threw {e}, switching...", flush=True)
            if OLLAMA_KEYS:
                current_ollama_key_index = (current_ollama_key_index + 1) % len(OLLAMA_KEYS)
            continue

    # 1. Remove block tags + their content (model internal reasoning)
    block_tags = r'<(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>.*?</(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>'
    answer = re.sub(block_tags, '', answer, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Remove any remaining XML/HTML-style tags but keep their text content
    answer = re.sub(r'<[^>]+>', '', answer).strip()

    if not answer:
        answer = "عذراً، لم أتمكن من إنشاء إجابة. حاول مرة أخرى."

    try:
        await db(lambda: supabase.table(history_table).insert({
            user_column: user_id,
            "role": "assistant",
            "content": answer
        }).execute())
    except Exception as e:
        print("Failed to save assistant history:", e, flush=True)

    return answer


# --- Process Telegram (inline - returns response dict for webhook reply) ---
async def process_telegram_inline(chat_id, text, msg_info):
    """Process Telegram message and return a webhook response dict.
    Since api.telegram.org is blocked from HF Spaces, we return the reply
    directly in the webhook response body (Telegram supports this)."""
    print(f"[TG] >>> Processing msg from {chat_id}: '{text[:50]}'", flush=True)
    try:
        # Check for contact sharing
        if "contact" in msg_info:
            phone = msg_info["contact"].get("phone_number")
            first_name = msg_info.get("from", {}).get("first_name", "")
            username = msg_info.get("from", {}).get("username", "")

            try:
                await db(lambda: supabase.table("bot_users").upsert({
                    "chat_id": chat_id,
                    "phone_number": phone,
                    "name": first_name,
                    "username": username
                }).execute())
            except Exception as e:
                print("Failed to save user:", e, flush=True)

            print(f"[TG] Contact saved for {chat_id}, sending welcome", flush=True)
            return {
                "method": "sendMessage",
                "chat_id": chat_id,
                "text": "Hi welcome to Mr Maged's bot! 🎓\n\nHere you can feel free to ask any question you want and don't worry, I'm always here to help you 😊",
                "reply_markup": {"remove_keyboard": True}
            }

        if not text:
            return None

        # Check if user is registered and has phone
        print(f"[TG] Checking user registration for {chat_id}...", flush=True)
        user_check = await db(lambda: supabase.table("bot_users").select("phone_number").eq("chat_id", chat_id).execute())
        has_phone = len(user_check.data) > 0 and user_check.data[0].get("phone_number") is not None
        print(f"[TG] User {chat_id} has_phone={has_phone}", flush=True)

        if text == "/start":
            if not has_phone:
                keyboard = {
                    "keyboard": [[{"text": "مشاركة رقم الهاتف 📱", "request_contact": True}]],
                    "resize_keyboard": True,
                    "one_time_keyboard": True
                }
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "Hi welcome to Mr Maged's bot! 🎓\n\nHere you can feel free to ask any question you want and don't worry, I'm always here to help you 😊\n\nPlease share your phone number first by pressing the button below:",
                    "reply_markup": keyboard
                }
            else:
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "Hi welcome back! 🎓\n\nFeel free to ask me anything about English, I'm always here to help you 😊"
                }

        if not has_phone:
            keyboard = {
                "keyboard": [[{"text": "مشاركة رقم الهاتف 📱", "request_contact": True}]],
                "resize_keyboard": True,
                "one_time_keyboard": True
            }
            return {
                "method": "sendMessage",
                "chat_id": chat_id,
                "text": "عفواً، لازم تشارك رقم التليفون الأول عشان أقدر أجاوبك.",
                "reply_markup": keyboard
            }

        # /review command
        if text.startswith("/review"):
            review_text = text[7:].strip()
            if not review_text:
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "📝 To leave a review, type:\n/review followed by your feedback\n\nExample:\n/review The bot is very helpful!"
                }
            try:
                first_name = msg_info.get("from", {}).get("first_name", "")
                await db(lambda: supabase.table("reviews").insert({
                    "platform": "telegram",
                    "user_id": str(chat_id),
                    "user_name": first_name,
                    "review": review_text
                }).execute())
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "Thank you so much for your feedback! 🙏😊\nYour review has been saved successfully ✅"
                }
            except Exception as e:
                print(f"Failed to save review: {e}", flush=True)
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "Sorry, something went wrong. Please try again later."
                }

        # Admin commands
        if text.startswith("/") and text not in ["/start"]:
            if chat_id not in ADMIN_CHAT_IDS:
                return {
                    "method": "sendMessage",
                    "chat_id": chat_id,
                    "text": "عفواً، ليس لديك صلاحية لاستخدام هذا الأمر."
                }

        if text == "/restore":
            # Start sync in background, return immediate confirmation
            asyncio.create_task(run_sync_logic(chat_id))
            return {
                "method": "sendMessage",
                "chat_id": chat_id,
                "text": "⏳ جاري بدء عملية المزامنة... سيتم إبلاغك عند الانتهاء."
            }

        if text == "/files":
            try:
                result = await db(lambda: supabase.table("documents").select("metadata").execute())
                sources = set()
                for d in result.data:
                    meta = d.get('metadata', {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except:
                            meta = {}
                    if isinstance(meta, dict) and 'source' in meta:
                        sources.add(meta['source'])

                if sources:
                    files_msg = "📚 الملفات الموجودة حالياً في الذاكرة:\n\n" + "\n".join([f"🔹 {s}" for s in sorted(sources)])
                else:
                    files_msg = "مفيش ملفات موجودة حالياً."

                return {"method": "sendMessage", "chat_id": chat_id, "text": files_msg}
            except Exception as e:
                print("Failed to fetch files:", e, flush=True)
                return {"method": "sendMessage", "chat_id": chat_id, "text": "عذراً، حصل مشكلة في جلب قائمة الملفات."}

        # Normal message — get RAG response
        print(f"[TG] Getting RAG response for {chat_id}...", flush=True)
        answer = await get_rag_response(chat_id, text, "chat_history", "chat_id")
        print(f"[TG] Got answer ({len(answer)} chars), returning inline to {chat_id}", flush=True)
        return {"method": "sendMessage", "chat_id": chat_id, "text": answer}

    except Exception as e:
        print(f"[TG] !!! Processing error for {chat_id}: {e}", flush=True)
        return {
            "method": "sendMessage",
            "chat_id": chat_id,
            "text": f"عذراً، حصل خطأ تقني. حاول تاني."
        }


# --- Message Deduplication Cache ---
_processed_msg_ids = set()
_MAX_CACHE = 500


def _is_duplicate(msg_id):
    """Check if message was already processed. Returns True if duplicate."""
    global _processed_msg_ids
    if msg_id in _processed_msg_ids:
        return True
    _processed_msg_ids.add(msg_id)
    # Trim cache to prevent memory leak
    if len(_processed_msg_ids) > _MAX_CACHE:
        _processed_msg_ids = set(list(_processed_msg_ids)[-200:])
    return False


async def process_whatsapp_message(data):
    """Process WhatsApp webhook per Meta docs - filter by field, deduplicate, handle messages only."""
    try:
        if data.get("object") != "whatsapp_business_account":
            return

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                # Per Meta docs: only process "messages" field, skip everything else
                field = change.get("field")
                if field != "messages":
                    print(f"[WA] Skipping non-message webhook (field={field})", flush=True)
                    continue

                value = change.get("value", {})

                # Log failed delivery statuses (debug only)
                if "statuses" in value:
                    for status in value.get("statuses", []):
                        if status.get("status") == "failed":
                            print(f"[WA STATUS FAILED] to {status.get('recipient_id')}: {status.get('errors')}", flush=True)
                    # If this webhook ONLY has statuses and no messages, skip
                    if "messages" not in value:
                        return

                if "messages" not in value:
                    return

                messages = value["messages"]
                contacts = value.get("contacts", [])

                for msg in messages:
                    msg_id = msg.get("id")

                    # Deduplication: Meta retries webhooks, skip already-processed messages
                    if msg_id and _is_duplicate(f"wa_{msg_id}"):
                        print(f"[WA] Skipping duplicate msg {msg_id}", flush=True)
                        continue

                    if msg.get("type") != "text":
                        print(f"[WA] Skipping non-text msg type={msg.get('type')}", flush=True)
                        continue

                    phone = msg.get("from")
                    text = msg.get("text", {}).get("body", "")
                    contact_name = contacts[0].get("profile", {}).get("name", "User") if contacts else "User"

                    print(f"[WA] Processing message from {phone}: {text[:50]}", flush=True)

                    # Mark as read (blue ticks) - non-blocking
                    asyncio.create_task(mark_whatsapp_read(msg_id))

                    # Save user
                    try:
                        await db(lambda p=phone, cn=contact_name: supabase.table("whatsapp_users").upsert(
                            {"phone_number": p, "name": cn},
                            on_conflict="phone_number"
                        ).execute())
                    except Exception as e:
                        print("Failed to save WA user:", e, flush=True)

                    # Handle /review command
                    if text.strip().lower().startswith("/review"):
                        review_text = text[7:].strip()
                        if not review_text:
                            await send_whatsapp(phone, "📝 To leave a review, send:\n/review followed by your feedback\n\nExample:\n/review The bot is very helpful!")
                        else:
                            try:
                                await db(lambda p=phone, cn=contact_name, rt=review_text: supabase.table("reviews").insert({
                                    "platform": "whatsapp",
                                    "user_id": p,
                                    "user_name": cn,
                                    "review": rt
                                }).execute())
                                await send_whatsapp(phone, "Thank you so much for your feedback! 🙏😊\nYour review has been saved successfully ✅")
                            except Exception as e:
                                print(f"Failed to save WA review: {e}", flush=True)
                                await send_whatsapp(phone, "Sorry, something went wrong. Please try again later.")
                        continue

                    # Check if first message (welcome)
                    try:
                        wa_history = await db(lambda p=phone: supabase.table("whatsapp_chat_history").select("id").eq("phone_number", p).limit(1).execute())
                        if not wa_history.data:
                            await send_whatsapp(phone, "Hi welcome to Mr Maged's bot! 🎓\n\nHere you can feel free to ask any question you want and don't worry, I'm always here to help you 😊")
                    except:
                        pass

                    # Get RAG response and send
                    answer = await get_rag_response(phone, text, "whatsapp_chat_history", "phone_number")
                    print(f"[WA] Got answer ({len(answer)} chars), sending to {phone}...", flush=True)
                    await send_whatsapp(phone, answer)
                    print(f"[WA] <<< Done for {phone}", flush=True)

    except Exception as e:
        print(f"[WA] Processing error: {e}", flush=True)


# --- Routes ---
@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()

        # Telegram deduplication using update_id
        update_id = data.get("update_id")
        if update_id and _is_duplicate(f"tg_{update_id}"):
            print(f"[TG] Skipping duplicate update {update_id}", flush=True)
            return JSONResponse({"ok": True})

        msg = data.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "")

        if not chat_id:
            return JSONResponse({"ok": True})

        # Process inline and return reply in webhook response body
        # (api.telegram.org is blocked from HF Spaces, so we use this Telegram-supported method)
        reply = await process_telegram_inline(chat_id, text, msg)
        if reply:
            print(f"[TG] Returning inline reply for {chat_id}", flush=True)
            return JSONResponse(reply)

        return JSONResponse({"ok": True})
    except Exception as e:
        print(f"Webhook error: {e}", flush=True)
        return JSONResponse({"ok": True})


@app.get("/whatsapp-webhook")
async def verify_whatsapp_webhook(request: Request):
    """Webhook verification for Meta."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            return PlainTextResponse(content=challenge)
    return JSONResponse({"status": "error"}, status_code=403)


@app.post("/whatsapp-webhook")
async def receive_whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        # Add to queue for sequential processing (stable, no race conditions)
        await _wa_queue.put(data)
        return JSONResponse({"ok": True})
    except Exception as e:
        print(f"WA Webhook error: {e}", flush=True)
        return JSONResponse({"ok": True})


@app.get("/sync-now")
async def sync_now():
    """Endpoint for Cron Job - syncs with embeddings via HF API."""
    try:
        report = await run_sync_logic()
        return {"status": "success", "message": report}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/set-webhook")
async def set_webhook(request: Request):
    base_url = str(request.base_url).rstrip("/")
    r = await _client.get(f"{TELEGRAM_API}/setWebhook?url={base_url}/webhook")
    return r.json()


@app.get("/")
async def home():
    return {"status": "running", "mode": "vector_search_async"}
