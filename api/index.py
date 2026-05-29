"""
Telegram RAG Bot - Ollama Cloud + Vector Search.
Uses Hugging Face Inference API (free) for embeddings.
Uses Supabase pgvector for similarity search.
Uses Ollama Cloud for chat responses.
"""
import os
import io
import json
import time
import sys
import requests
from fastapi import FastAPI, Request, BackgroundTasks
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

# --- Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "maged_bot_secure_token")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "").strip().replace('\ufeff', '').replace('\r', '').replace('\n', '')
GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]
current_gemini_key_index = 0
FOLDER_ID = "1xBpOgVa6gDT2MyZngiygXTYODC3LfvUu"
ADMIN_CHAT_IDS = [8284113566, 5103350500]

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Embedding via Gemini API ---
def get_embedding(text):
    """Get 768-dim embedding using Gemini API with failover."""
    global current_gemini_key_index
    if not GEMINI_KEYS:
        raise ValueError("No Gemini API keys found in .env (GEMINI_API_KEYS)")
        
    for attempt in range(len(GEMINI_KEYS)):
        current_key = GEMINI_KEYS[current_gemini_key_index]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={current_key}"
        
        try:
            resp = requests.post(url, json={
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [{"text": text}]
                },
                "outputDimensionality": 768
            }, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                return data["embedding"]["values"]
            else:
                print(f"Key {current_gemini_key_index} failed with {resp.status_code}: {resp.text}")
                # Switch to next key
                current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_KEYS)
        except Exception as e:
            print(f"Key {current_gemini_key_index} network error: {e}")
            current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_KEYS)
            
    raise Exception("All Gemini API keys failed.")

def get_query_embedding(text):
    """Get embedding for a query."""
    return get_embedding(text)


# --- Vector Search in Supabase ---
def search_similar(query, limit=5):
    """Search for similar documents using vector similarity."""
    embedding = get_query_embedding(query)
    
    result = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_threshold": 0.3,
        "match_count": limit
    }).execute()
    
    return result.data if result.data else []


# --- Sync Logic (for Cron Job) ---
def get_drive_service():
    service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    info = None
    if service_account_info:
        try:
            info = json.loads(service_account_info)
        except Exception as e:
            print("Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON:", e)
            
    if info:
        creds = service_account.Credentials.from_service_account_info(info)
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "service-account.json")
        creds = service_account.Credentials.from_service_account_file(path)
    
    return build('drive', 'v3', credentials=creds.with_scopes(
        ['https://www.googleapis.com/auth/drive.readonly']))


def get_text_embedding_hf(text):
    """Legacy function wrapper (kept for compatibility)."""
    return get_embedding(text)


def run_sync_logic(chat_id=None):
    if chat_id:
        send_telegram(chat_id, "بدأ عملية المزامنة (Restore)... جاري فحص الملفات.")
        
    service = get_drive_service()
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute()
    files = results.get('files', [])
    
    if not files:
        if chat_id:
            send_telegram(chat_id, "لم يتم العثور على أي ملفات PDF.")
        return "No PDF files found."

    # Fetch already synced files
    synced_response = supabase.table("synced_files").select("file_id").execute()
    synced_file_ids = {row["file_id"] for row in synced_response.data}
    
    files_to_sync = [f for f in files if f['id'] not in synced_file_ids]
    
    if not files_to_sync:
        if chat_id:
            send_telegram(chat_id, "كل الملفات تم عمل sync لها مسبقاً. مفيش ملفات جديدة.")
        return "All files already synced."

    if chat_id:
        send_telegram(chat_id, f"تم إيجاد {len(files_to_sync)} ملف جديد. جاري التحويل...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    total_chunks = 0

    for file_info in files_to_sync:
        if chat_id:
            send_telegram(chat_id, f"جاري معالجة: {file_info['name']}")
            
        request = service.files().get_media(fileId=file_info['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        fh.seek(0)
        reader = PdfReader(fh)
        text = "".join([p.extract_text() or "" for p in reader.pages])
        
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            supabase.table("documents").insert({
                "content": chunk,
                "embedding": emb,
                "metadata": json.dumps({"source": file_info['name'], "chunk": i})
            }).execute()
            total_chunks += 1
            
        # Mark as synced
        supabase.table("synced_files").insert({
            "file_id": file_info['id'],
            "file_name": file_info['name']
        }).execute()
            
    if chat_id:
        send_telegram(chat_id, f"تم الانتهاء بنجاح! تمت معالجة {len(files_to_sync)} ملف ({total_chunks} أجزاء).")
        
    return f"Synced {len(files_to_sync)} files ({total_chunks} chunks)."


# --- Bot Logic ---
def ask_ollama(system_prompt, user_message):
    response = requests.post(
        "https://ollama.com/api/chat",
        headers={"Authorization": f"Bearer {OLLAMA_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        },
        timeout=120
    )
    return response.json().get("message", {}).get("content", "Sorry, I couldn't process that.")


# --- Robust HTTP Session with retries for HF Spaces ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _make_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

http_session = _make_session()


def send_telegram(chat_id, text):
    for i in range(0, len(text), 4000):
        try:
            http_session.post(f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": text[i:i+4000]}, timeout=60)
        except Exception as e:
            print(f"[TG SEND FAIL] {e}")

def send_telegram_typing(chat_id):
    try:
        http_session.post(f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"}, timeout=15)
    except:
        pass

def send_telegram_keyboard(chat_id, text, keyboard):
    try:
        http_session.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "reply_markup": keyboard}, 
            timeout=60)
    except Exception as e:
        print(f"[TG KB FAIL] {e}")

def mark_whatsapp_read(message_id):
    if not WHATSAPP_PHONE_ID or not WHATSAPP_TOKEN:
        return
    try:
        url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_ID}/messages"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
        http_session.post(url, headers=headers, json={
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }, timeout=15)
    except:
        pass

def send_whatsapp(to_phone, text):
    if not WHATSAPP_PHONE_ID or not WHATSAPP_TOKEN:
        print("WhatsApp credentials missing.")
        return
    url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    for i in range(0, len(text), 4000):
        data = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {
                "body": text[i:i+4000]
            }
        }
        try:
            r = http_session.post(url, headers=headers, json=data, timeout=60)
            print(f"[WA SEND] Status: {r.status_code} | Response: {r.text}")
        except Exception as e:
            print(f"Failed to send WA message: {e}")

GREETINGS = {"hi", "hello", "hey", "هلا", "اهلا", "مرحبا", "هاي", "السلام عليكم", "ازيك", "ازيكم", "صباح الخير", "مساء الخير", "يا مستر", "مستر"}

def get_rag_response(user_id, text, history_table, user_column):
    # Smart: skip RAG search for greetings/short messages
    text_lower = text.strip().lower()
    is_greeting = text_lower in GREETINGS or len(text_lower) < 4
    
    if is_greeting:
        similar_docs = []
        print(f"[SMART] Skipped RAG for greeting: '{text_lower}'")
    else:
        # Use fewer docs for short questions, more for complex ones
        doc_limit = 5 if len(text) < 30 else 10
        similar_docs = search_similar(text, limit=doc_limit)
        print(f"[SMART] RAG search with {doc_limit} docs for: '{text[:40]}'")


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
        result = supabase.table("documents").select("content, metadata").limit(20).execute()
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
- If the student writes in Arabic → reply in Egyptian Arabic (عامية مصرية) BUT naturally mix in English words and short phrases (A2-B1 level). You are an English tutor, so ALWAYS weave English into your Arabic replies to help the student learn. Examples:
  "بص يا بطل، الـ present simple بنستخدمه for habits and routines يعني حاجات بتحصل always"
  "الكلمة دي meaning بتاعها هي..."
  "Try to think about it كده... لو عايز تقول إنك بتعمل حاجة every day بتستخدم..."
- If the student writes in English → reply ENTIRELY in English. Be friendly, natural, and encouraging.
- Grammar terms and English vocabulary ALWAYS stay in English regardless of reply language.

MEMORY:
- You HAVE full conversation memory. The previous messages are REAL past messages with this specific student. 
- You MUST remember what was discussed. If the student says you talked before, ACKNOWLEDGE it. NEVER say you cannot remember.

PERSONALITY:
- Act exactly like MR Maged: relaxed, friendly, encouraging Egyptian English teacher.
- Give the core concept simply and directly. No textbook essays.
- Explain step by step using MR Maged's style and his exact words from the materials.

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
        history_response = supabase.table(history_table).select("role, content").eq(user_column, user_id).order("created_at", desc=True).limit(history_limit).execute()
        history = list(reversed(history_response.data))
        print(f"[HISTORY] Loaded {len(history)}/{history_limit} messages for {user_id}")
    except Exception as e:
        print("Failed to fetch chat history:", e)

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": text})

    try:
        supabase.table(history_table).insert({
            user_column: user_id,
            "role": "user",
            "content": text
        }).execute()
    except Exception as e:
        print("Failed to save user history:", e)

    response = http_session.post(
        "https://ollama.com/api/chat",
        json={
            "model": "gpt-oss:120b",
            "messages": messages,
            "stream": False
        },
        headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
        timeout=120
    )
    import re
    answer = response.json().get("message", {}).get("content", "Sorry, I couldn't generate an answer.")
    
    # 1. Remove block tags + their content (model internal reasoning)
    block_tags = r'<(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>.*?</(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>'
    answer = re.sub(block_tags, '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 2. Remove any remaining XML/HTML-style tags but keep their text content
    answer = re.sub(r'<[^>]+>', '', answer).strip()
    
    if not answer:
        answer = "عذراً، لم أتمكن من إنشاء إجابة. حاول مرة أخرى."
    
    try:
        supabase.table(history_table).insert({
            user_column: user_id,
            "role": "assistant",
            "content": answer
        }).execute()
    except Exception as e:
        print("Failed to save assistant history:", e)

    return answer


# --- Routes ---
def process_telegram_message(chat_id, text, msg_info):
    print(f"[TG] >>> Processing msg from {chat_id}: '{text[:50]}'", flush=True)
    try:
        # Check for contact sharing
        if "contact" in msg_info:
            phone = msg_info["contact"].get("phone_number")
            first_name = msg_info.get("from", {}).get("first_name", "")
            username = msg_info.get("from", {}).get("username", "")
            
            try:
                supabase.table("bot_users").upsert({
                    "chat_id": chat_id,
                    "phone_number": phone,
                    "name": first_name,
                    "username": username
                }).execute()
            except Exception as e:
                print("Failed to save user:", e)
            
            remove_kb = {"remove_keyboard": True}
            send_telegram_keyboard(chat_id, "تم تأكيد رقم تليفونك بنجاح! البوت متاح ليك دلوقتي، وتقدر تسأل أي سؤال في المنهج.", remove_kb)
            return

        if not text:
            return

        # Check if user is registered and has phone
        user_check = supabase.table("bot_users").select("phone_number").eq("chat_id", chat_id).execute()
        has_phone = len(user_check.data) > 0 and user_check.data[0].get("phone_number") is not None

        if text == "/start":
            if not has_phone:
                keyboard = {
                    "keyboard": [[{"text": "مشاركة رقم الهاتف 📱", "request_contact": True}]],
                    "resize_keyboard": True,
                    "one_time_keyboard": True
                }
                send_telegram_keyboard(chat_id, 
                    "أهلاً بيك! أنا مساعد مستر ماجد.\nعشان أقدر أساعدك، يرجى الضغط على الزرار بالأسفل لمشاركة رقم هاتفك:", 
                    keyboard)
            else:
                send_telegram(chat_id, "أهلاً بك مجدداً! تفضل اسألني في المنهج.")
            return

        if not has_phone:
            keyboard = {
                "keyboard": [[{"text": "مشاركة رقم الهاتف 📱", "request_contact": True}]],
                "resize_keyboard": True,
                "one_time_keyboard": True
            }
            send_telegram_keyboard(chat_id, "عفواً، لازم تشارك رقم التليفون الأول عشان أقدر أجاوبك.", keyboard)
            return

        # Restrict all commands (except /start) to admins only
        if text.startswith("/") and text != "/start":
            if chat_id not in ADMIN_CHAT_IDS:
                send_telegram(chat_id, "عفواً، ليس لديك صلاحية لاستخدام هذا الأمر.")
                return

        if text == "/restore":
            run_sync_logic(chat_id)
            return

        if text == "/files":
            try:
                result = supabase.table("documents").select("metadata").execute()
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
                
                send_telegram(chat_id, files_msg)
            except Exception as e:
                print("Failed to fetch files:", e)
                send_telegram(chat_id, "عذراً، حصل مشكلة في جلب قائمة الملفات.")
            return

        # Send typing indicator then process
        send_telegram_typing(chat_id)
        print(f"[TG] Getting RAG response for {chat_id}...", flush=True)
        answer = get_rag_response(chat_id, text, "chat_history", "chat_id")
        print(f"[TG] Got answer ({len(answer)} chars), sending to {chat_id}...", flush=True)
        send_telegram(chat_id, answer)
        print(f"[TG] <<< Done for {chat_id}", flush=True)

    except Exception as e:
        print(f"[TG] !!! Processing error for {chat_id}: {e}", flush=True)
        try:
            send_telegram(chat_id, f"عذراً، حصل خطأ تقني: {str(e)[:200]}")
        except Exception as e2:
            print(f"[TG] !!! Failed to send error msg too: {e2}", flush=True)

def process_whatsapp_message(data):
    print(f"[WA] >>> Processing incoming WA data", flush=True)
    try:
        if "object" in data and data["object"] == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    
                    if "messages" in value:
                        messages = value["messages"]
                        contacts = value.get("contacts", [])
                        
                        for msg in messages:
                            if msg.get("type") != "text":
                                continue
                                
                            phone = msg.get("from")
                            text = msg.get("text", {}).get("body", "")
                            msg_id = msg.get("id")
                            
                            contact_name = contacts[0].get("profile", {}).get("name", "User") if contacts else "User"

                            # Mark as read (blue ticks) immediately
                            mark_whatsapp_read(msg_id)

                            try:
                                supabase.table("whatsapp_users").upsert(
                                    {"phone_number": phone, "name": contact_name},
                                    on_conflict="phone_number"
                                ).execute()
                            except Exception as e:
                                print("Failed to save WA user:", e)

                            print(f"[WA] Processing message from {phone}: {text[:50]}")
                            answer = get_rag_response(phone, text, "whatsapp_chat_history", "phone_number")
                            print(f"[WA] Got answer, sending to {phone}...")
                            send_whatsapp(phone, answer)
                            print(f"[WA] send_whatsapp completed for {phone}")
    except Exception as e:
        print(f"WA processing error: {e}")

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        msg = data.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "")

        if not chat_id:
            return JSONResponse({"ok": True})
            
        # Process message in background to avoid blocking event loop and timeout
        background_tasks.add_task(process_telegram_message, chat_id, text, msg)
        return JSONResponse({"ok": True})
    except Exception as e:
        print(f"Webhook error: {e}")
        return JSONResponse({"status": "error", "detail": str(e)})

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
async def receive_whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        background_tasks.add_task(process_whatsapp_message, data)
        return JSONResponse({"ok": True})
    except Exception as e:
        print(f"WA Webhook error: {e}")
        return JSONResponse({"status": "error", "detail": str(e)})


@app.get("/sync-now")
async def sync_now():
    """Endpoint for Cron Job - syncs with embeddings via HF API."""
    try:
        report = run_sync_logic()
        return {"status": "success", "message": report}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/set-webhook")
async def set_webhook(request: Request):
    base_url = str(request.base_url).rstrip("/")
    r = http_session.get(f"{TELEGRAM_API}/setWebhook?url={base_url}/webhook", timeout=60)
    return r.json()


@app.get("/")
async def home():
    return {"status": "running", "mode": "vector_search"}
