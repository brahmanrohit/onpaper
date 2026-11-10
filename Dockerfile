# OnPaper — production image (Hugging Face Spaces, or any Docker host).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hosted: AI is Groq only, so never spend startup time probing for Ollama.
    AI_BACKEND=groq

WORKDIR /app

# Dependencies + NLTK data baked into the image, so cold starts download nothing.
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -m nltk.downloader -q -d /usr/share/nltk_data punkt_tab punkt stopwords

COPY . .
# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4)"

# Streamlit's XSRF and CORS protections stay on (their defaults).
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
