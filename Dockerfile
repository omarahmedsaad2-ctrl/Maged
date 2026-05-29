FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Force Python to flush logs immediately
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces require running on port 7860
CMD ["sh", "-c", "echo '===== Application Startup at '$(date -u '+%Y-%m-%d %H:%M:%S')' =====' && uvicorn api.index:app --host 0.0.0.0 --port 7860"]
