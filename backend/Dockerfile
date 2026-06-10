# Briefr backend image — serves both the FastAPI app (default CMD) and the
# Celery worker (compose overrides the command).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch FIRST: the default Linux wheels pull multi-gigabyte CUDA
# dependencies that this stack never uses (cross-encoder runs on CPU).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake runtime assets into the image so the first request isn't stalled by
# downloads: nltk sentence tokenizer + the two local models (semantic-chunker
# embedder and cross-encoder reranker — must match the config.py defaults).
RUN python -m nltk.downloader punkt punkt_tab && \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

EXPOSE 8000
# --reload pairs with the compose bind-mount of ./backend for live development.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
