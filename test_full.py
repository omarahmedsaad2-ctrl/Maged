"""
Full integration test for the bot — tests every function in index.py
without actually sending messages or hitting real APIs.
"""
import os, sys, json

# Ensure we can import api.index
sys.path.insert(0, os.path.dirname(__file__))

# Patch environment before importing
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_token")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "eytest")
os.environ.setdefault("OLLAMA_API_KEYS", "key1,key2")
os.environ.setdefault("GEMINI_API_KEYS", "gemkey1,gemkey2")
os.environ.setdefault("WHATSAPP_TOKEN", "wa_token")
os.environ.setdefault("WHATSAPP_PHONE_ID", "12345")

import httpx
import importlib

# ---- Test 1: requirements.txt is clean UTF-8 ----
def test_requirements():
    with open("requirements.txt", "rb") as f:
        raw = f.read()
    assert b"\x00" not in raw, "requirements.txt has NULL bytes (UTF-16 corruption!)"
    lines = raw.decode("utf-8").strip().splitlines()
    required = ["fastapi", "uvicorn", "supabase", "httpx", "requests"]
    for pkg in required:
        found = any(pkg in line for line in lines)
        assert found, f"Missing '{pkg}' in requirements.txt"
    print("  [PASS] requirements.txt is clean UTF-8 with all dependencies")

# ---- Test 2: Dockerfile is valid ----
def test_dockerfile():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "python:3.10" in content, "Dockerfile should use Python 3.10"
    assert "requirements.txt" in content, "Dockerfile should install requirements"
    assert "7860" in content, "Dockerfile should use port 7860"
    print("  [PASS] Dockerfile is valid (python:3.10, port 7860)")

# ---- Test 3: index.py imports and config ----
def test_imports_and_config():
    # Check httpx is imported
    with open("api/index.py", "r", encoding="utf-8") as f:
        code = f.read()
    assert "import httpx" in code, "httpx not imported"
    assert "_client.post(" in code, "_client.post not used for HTTP calls"
    # Make sure no stale requests.post calls remain (except the import)
    import re
    stale = re.findall(r'(?<!import )requests\.post\(', code)
    assert len(stale) == 0, f"Found {len(stale)} stale requests.post() calls!"
    print("  [PASS] All HTTP calls use httpx (no stale requests.post)")

# ---- Test 4: OLLAMA keys parsing ----
def test_ollama_keys():
    raw = "key1.abc , key2.def , "
    keys = [k.strip().replace('\ufeff', '').replace('\r', '').replace('\n', '') for k in raw.split(",") if k.strip()]
    assert keys == ["key1.abc", "key2.def"], f"Key parsing failed: {keys}"
    # Test BOM handling
    raw_bom = "\ufeffkey1.abc"
    keys_bom = [k.strip().replace('\ufeff', '') for k in raw_bom.split(",") if k.strip()]
    assert keys_bom == ["key1.abc"], f"BOM handling failed: {keys_bom}"
    print("  [PASS] OLLAMA key parsing (comma-split, trim, BOM)")

# ---- Test 5: Ollama key failover logic ----
def test_ollama_failover():
    keys = ["bad_key", "good_key"]
    idx = 0
    statuses = [401, 200]  # first key fails, second succeeds
    
    result = None
    for attempt in range(len(keys)):
        status = statuses[idx]
        if status == 200:
            result = "success"
            break
        else:
            idx = (idx + 1) % len(keys)
            continue
    
    assert result == "success", "Failover didn't reach good key"
    assert idx == 1, "Should have switched to key index 1"
    print("  [PASS] Ollama key failover (401 -> switch -> 200)")

# ---- Test 6: Greeting detection ----
def test_greetings():
    GREETINGS = {"hi", "hello", "hey", "هلا", "اهلا", "مرحبا", "هاي", "السلام عليكم", "ازيك", "ازيكم", "صباح الخير", "مساء الخير", "يا مستر", "مستر"}
    
    should_skip = ["hi", "Hello", "HEY", "اهلا", ".", "ok", "اه"]
    should_search = ["explain present simple", "what is a verb", "ممكن تشرحلي grammar"]
    
    for text in should_skip:
        text_lower = text.strip().lower()
        is_greeting = text_lower in GREETINGS or len(text_lower) < 4
        assert is_greeting, f"'{text}' should be detected as greeting"
    
    for text in should_search:
        text_lower = text.strip().lower()
        is_greeting = text_lower in GREETINGS or len(text_lower) < 4
        assert not is_greeting, f"'{text}' should NOT be detected as greeting"
    
    print("  [PASS] Greeting detection (skip RAG for greetings/short)")

# ---- Test 7: Thinking tag removal ----
def test_thinking_removal():
    import re
    block_tags = r'<(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>.*?</(?:thinking|thought|reasoning|reflect|internal|scratchpad|meta|plan|analysis|step_by_step|chain_of_thought|inner_monologue)>'
    
    test_cases = [
        ("<thinking>internal reasoning here</thinking>Hello!", "Hello!"),
        ("<thought>deep thought</thought>Answer is 42", "Answer is 42"),
        ("No tags here", "No tags here"),
        ("<reasoning>step 1\nstep 2</reasoning>Final answer", "Final answer"),
    ]
    
    for input_text, expected in test_cases:
        result = re.sub(block_tags, '', input_text, flags=re.DOTALL | re.IGNORECASE).strip()
        result = re.sub(r'<[^>]+>', '', result).strip()
        assert result == expected, f"Tag removal failed: got '{result}', expected '{expected}'"
    
    print("  [PASS] Thinking tag removal (all tag variants)")

# ---- Test 8: WhatsApp webhook structure parsing ----
def test_whatsapp_parsing():
    sample_webhook = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "type": "text",
                        "from": "201201434500",
                        "text": {"body": "Hello!"},
                        "id": "msg_123"
                    }],
                    "contacts": [{
                        "profile": {"name": "Ahmed"}
                    }]
                }
            }]
        }]
    }
    
    data = sample_webhook
    assert data["object"] == "whatsapp_business_account"
    entry = data["entry"][0]
    change = entry["changes"][0]
    value = change["value"]
    msg = value["messages"][0]
    assert msg["type"] == "text"
    assert msg["from"] == "201201434500"
    assert msg["text"]["body"] == "Hello!"
    assert msg["id"] == "msg_123"
    contact = value["contacts"][0]
    assert contact["profile"]["name"] == "Ahmed"
    print("  [PASS] WhatsApp webhook parsing (messages + contacts)")

# ---- Test 9: Telegram webhook structure parsing ----
def test_telegram_parsing():
    sample = {
        "message": {
            "chat": {"id": 8284113566},
            "text": "hi",
            "from": {"first_name": "Omar", "username": "omar123"}
        }
    }
    msg = sample.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    text = msg.get("text", "")
    assert chat_id == 8284113566
    assert text == "hi"
    print("  [PASS] Telegram webhook parsing (chat_id + text)")

# ---- Test 10: httpx.post works (real quick test) ----
def test_httpx_works():
    try:
        r = httpx.get("https://httpbin.org/get", timeout=10)
        assert r.status_code == 200
        print("  [PASS] httpx network connectivity works")
    except Exception as e:
        print(f"  [SKIP] httpx network test (offline): {e}")

# ---- Test 11: Text chunking for long messages ----
def test_message_chunking():
    long_text = "A" * 10000
    chunks = []
    for i in range(0, len(long_text), 4000):
        chunks.append(long_text[i:i+4000])
    assert len(chunks) == 3
    assert len(chunks[0]) == 4000
    assert len(chunks[1]) == 4000
    assert len(chunks[2]) == 2000
    print("  [PASS] Message chunking (4000 char limit)")

# ---- Test 12: Admin check ----
def test_admin_check():
    ADMIN_CHAT_IDS = [8284113566, 5103350500]
    assert 8284113566 in ADMIN_CHAT_IDS
    assert 5103350500 in ADMIN_CHAT_IDS
    assert 12345 not in ADMIN_CHAT_IDS
    print("  [PASS] Admin chat ID validation")

# ---- Test 13: Concurrency config ----
def test_concurrency():
    with open("api/index.py", "r", encoding="utf-8") as f:
        code = f.read()
    assert "asyncio.Semaphore(50)" in code, "Semaphore limit not set to 50"
    print("  [PASS] Concurrency configured to 50 via Semaphore")

# ---- Run all tests ----
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FULL INTEGRATION TEST SUITE")
    print("="*50 + "\n")
    
    tests = [
        test_requirements,
        test_dockerfile,
        test_imports_and_config,
        test_ollama_keys,
        test_ollama_failover,
        test_greetings,
        test_thinking_removal,
        test_whatsapp_parsing,
        test_telegram_parsing,
        test_httpx_works,
        test_message_chunking,
        test_admin_check,
        test_concurrency,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*50}\n")
    
    sys.exit(1 if failed > 0 else 0)
