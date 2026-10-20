# For Hithub Spaces Deployment

FROM python:3.11-slim

# 1. Install minimal OS dependencies (git & C-compilers for Tree-Sitter; NO LibreOffice/Java bloat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Hugging Face Spaces requirement: Create non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

WORKDIR $HOME/app

# 3. Install Python dependencies first (cached layer)
COPY --chown=user:user requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Pre-warm FastEmbed BAAI/bge-small-en-v1.5 ONNX weights at build time
RUN python -c "from llama_index.embeddings.fastembed import FastEmbedEmbedding; FastEmbedEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# 5. Copy application source code and install in editable mode
COPY --chown=user:user . .
RUN pip install --no-cache-dir --no-deps -e . && \
    mkdir -p $HOME/app/data && \
    chmod +x $HOME/app/start.sh

# Hugging Face Spaces default port is 7860 (Streamlit UI); FastAPI runs internally on 8000
EXPOSE 7860

CMD ["./start.sh"]