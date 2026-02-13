FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
WORKDIR /app

# System deps:
# - espeak-ng: Kokoro docs recommend it for fallback / some languages
# - libsndfile1: required by python-soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates \
    espeak-ng \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.4 wheels)
RUN python -m pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Optional: keep model/cache on a volume
ENV HF_HOME=/models/hf
ENV XDG_CACHE_HOME=/models/cache

COPY app ./app

ENV KOKORO_LANG_CODE=a
ENV KOKORO_VOICE=af_heart
ENV KOKORO_SAMPLE_RATE=24000
ENV MAX_INFLIGHT_PER_WORKER=16

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
