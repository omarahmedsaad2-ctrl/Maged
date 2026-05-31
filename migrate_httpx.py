import re

with open(r"c:\Users\LILMAR\Desktop\Maged\ai-bot\api\index.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace import requests with import httpx
if "import httpx" not in code:
    code = code.replace("import requests", "import requests\nimport httpx")

# Replace requests.post with httpx.post
code = code.replace("requests.post(", "httpx.post(")

# There is a session object `http_session` used in `set_webhook`, I will change it to httpx.get
code = code.replace("http_session.get(", "httpx.get(")

# Note: requests.post json=data timeout=X is the SAME signature in httpx.post
# Exception handling: httpx raises httpx.RequestError instead of requests.exceptions.RequestException, but we use `Exception as e` everywhere, so it's fine.

with open(r"c:\Users\LILMAR\Desktop\Maged\ai-bot\api\index.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Rewrote requests.post to httpx.post successfully!")
