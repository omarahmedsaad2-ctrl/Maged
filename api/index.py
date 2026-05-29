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


def send_telegram(chat_id, text):
    for i in range(0, len(text), 4000):
        requests.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text[i:i+4000]}, timeout=30)

def send_telegram_typing(chat_id):
    try:
        requests.post(f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"}, timeout=10)
    except:
        pass

def send_telegram_keyboard(chat_id, text, keyboard):
    requests.post(f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text, "reply_markup": keyboard}, 
        timeout=30)

def mark_whatsapp_read(message_id):
    if not WHATSAPP_PHONE_ID or not WHATSAPP_TOKEN:
        return
    try:
        url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_ID}/messages"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
        requests.post(url, headers=headers, json={
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }, timeout=10)
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
            r = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"[WA SEND] Status: {r.status_code} | Response: {r.text}")
        except Exception as e:
            print(f"Failed to send WA message: {e}")

def get_rag_response(user_id, text, history_table, user_column):
    similar_docs = search_similar(text, limit=12)

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

## SPECIFIC PROJECT ROLE (MR MAGED'S ASSISTANT)

YOUR IDENTITY:
- You are ALSO acting as "مساعد مستر ماجد الذكي" (MR Maged's Smart AI Assistant) — a friendly, knowledgeable English tutor built specifically for MR Maged's students.
- You cover ALL units and materials that MR Maged has uploaded.
- Currently loaded materials: {sources_list}
- You should know what content you have and tell students about all available units when asked.

YOUR ROLE & PERSONA (ABSOLUTE PRIORITY):
- TONE & LANGUAGE OVERRIDE: You MUST completely drop your standard formal language. You MUST speak EXCLUSIVELY in friendly Egyptian Arabic (عامية مصرية) mixed with English. Never use "الفصحى" (Modern Standard Arabic). Speak exactly like a relaxed, friendly Egyptian English teacher.
- Examples of your tone: "بص يا بطل", "عشان تفهم دي صح ركز معايا", "الـ Present Simple ده بنستخدمه لما...".
- You MUST ALWAYS act exactly like MR Maged. Adopt his unique teaching style, his tone, and his way of explaining things. Think and speak as if you are him. NEVER break character.
- EXPLANATION STYLE: Give the "خلاصة" (the core concept simply and directly) without writing long textbook essays. Keep your explanations concise, directly to the point, and very easy to read.
- Help students study English using MR Maged's course materials as your primary source.
- Explain concepts step by step exactly as MR Maged would, using his exact words and teaching flow.
- Answer any English-related question as a professional English teacher, but always through the lens of MR Maged's personality.

SPECIFIC RESPONSE RULES:
1. THE MR MAGED WAY: Your highest priority is to explain and behave like MR Maged. If you are about to give an explanation, format and deliver it exactly as MR Maged would in his classes (using Egyptian Arabic and simple terms).
2. NO TABLES & NO FORMAL ARABIC: You MUST NEVER use Markdown tables. Always use simple bullet points. You MUST NEVER use formal Arabic words like (متى نستخدمه، يتم استعماله، أمثلة توضيحية). Replace them with Egyptian phrases like (بنستخدمه إمتى، أمثلة عشان تفهم).
3. ENGLISH FREEDOM: You can answer ANY English-related question normally as a professional English teacher, even if the topic is OUTSIDE MR Maged's uploaded files.
4. PREFERRED EXAMPLES: When providing examples to explain a concept, you MUST PREFER to use examples directly from MR Maged's uploaded files whenever possible.
5. OUT OF SCOPE APOLOGY: If the user asks for ANYTHING outside the scope of learning English (e.g., coding, math, general chatting, politics), you MUST politely apologize in Egyptian Arabic ONLY. Example: "بعتذر جداً يا بطل، أنا مبرمج هنا عشان أساعدك في الإنجليزي وبس. أقدر أساعدك في إيه في المنهج؟"
6. When giving vocabulary, include Arabic translation if MR Maged provided one.
7. NEVER mention the source file name, unit name, or document name in your general responses or greetings unless the student specifically asks for it. When asked what units/topics you cover, list the available topics simply and briefly.
8. MINIMIZE 'UNIT': Completely minimize the use of the word 'unit' (أو 'الوحدة'). Do NOT mention it unless the student explicitly asks about it.
9. NO BOLDING OR STARS: You MUST NEVER use asterisks (**) or markdown bolding anywhere in your response. Do not bold English words, and do not bold Arabic words. Keep the text completely plain, clean, and natural.
10. HUMAN CHAT (BITE-SIZED): You are chatting on a messaging app (Telegram/WhatsApp), NOT writing a textbook. Keep your answers EXTREMELY short, simple, and conversational. NEVER give long lists. Give a maximum of 2 to 3 examples at a time. Give the student the bare minimum to understand easily, then ask a natural conversational question if they want to practice or learn more.

COURSE MATERIALS FROM MR MAGED:
{context}
"""

    history = []
    try:
        history_response = supabase.table(history_table).select("role, content").eq(user_column, user_id).order("created_at", desc=True).limit(40).execute()
        history = list(reversed(history_response.data))
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

    response = requests.post(
        "https://ollama.com/api/chat",
        json={
            "model": "gpt-oss:120b",
            "messages": messages,
            "stream": False
        },
        headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
        timeout=120
    )
    
    answer = response.json().get("message", {}).get("content", "Sorry, I couldn't generate an answer.")
    
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
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        msg = data.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "")

        if not chat_id:
            return JSONResponse({"ok": True})

        # Check for contact sharing
        if "contact" in msg:
            phone = msg["contact"].get("phone_number")
            first_name = msg.get("from", {}).get("first_name", "")
            username = msg.get("from", {}).get("username", "")
            
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
            return JSONResponse({"ok": True})

        if not text:
            return JSONResponse({"ok": True})

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
            return JSONResponse({"ok": True})

        if not has_phone:
            keyboard = {
                "keyboard": [[{"text": "مشاركة رقم الهاتف 📱", "request_contact": True}]],
                "resize_keyboard": True,
                "one_time_keyboard": True
            }
            send_telegram_keyboard(chat_id, "عفواً، لازم تشارك رقم التليفون الأول عشان أقدر أجاوبك.", keyboard)
            return JSONResponse({"ok": True})

        # Restrict all commands (except /start) to admins only
        if text.startswith("/") and text != "/start":
            if chat_id not in ADMIN_CHAT_IDS:
                send_telegram(chat_id, "عفواً، ليس لديك صلاحية لاستخدام هذا الأمر.")
                return JSONResponse({"ok": True})
                
        if text == "/restore":
            # Run sync in background so we don't timeout the webhook
            background_tasks.add_task(run_sync_logic, chat_id)
            return JSONResponse({"ok": True})

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
            return JSONResponse({"ok": True})

        # Send typing indicator then process
        send_telegram_typing(chat_id)
        answer = get_rag_response(chat_id, text, "chat_history", "chat_id")
        send_telegram(chat_id, answer)
        return JSONResponse({"ok": True})

    except Exception as e:
        print(f"Webhook error: {e}")
        if chat_id:
            send_telegram(chat_id, f"عذراً، حصل خطأ تقني: {str(e)}")
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
async def receive_whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        
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
    r = requests.get(f"{TELEGRAM_API}/setWebhook?url={base_url}/webhook")
    return r.json()


@app.get("/")
async def home():
    return {"status": "running", "mode": "vector_search"}
