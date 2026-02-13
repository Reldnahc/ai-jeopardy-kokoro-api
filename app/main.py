import io
import os
import asyncio

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from kokoro import KPipeline

KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
SAMPLE_RATE = int(os.getenv("KOKORO_SAMPLE_RATE", "24000"))

MAX_INFLIGHT_PER_WORKER = int(os.getenv("MAX_INFLIGHT_PER_WORKER", "1"))
_sem = asyncio.Semaphore(MAX_INFLIGHT_PER_WORKER)

app = FastAPI(title="Kokoro Python API", version="1.0")

pipeline: KPipeline | None = None


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str | None = None


@app.on_event("startup")
def _load():
    global pipeline
    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)


@app.get("/health")
def health():
    return {
        "ok": True,
        "lang_code": KOKORO_LANG_CODE,
        "default_voice": DEFAULT_VOICE,
        "sample_rate": SAMPLE_RATE,
        "cuda": torch.cuda.is_available(),
        "max_inflight_per_worker": MAX_INFLIGHT_PER_WORKER,
    }


def synth_to_wav_bytes(text: str, voice: str) -> bytes:
    if pipeline is None:
        raise RuntimeError("pipeline not loaded")

    # Kokoro yields chunks; stitch into one array
    chunks: list[np.ndarray] = []
    for _i, (_gs, _ps, audio) in enumerate(pipeline(text, voice=voice)):
        # audio is typically a float numpy array
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("no audio generated")

    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    buf = io.BytesIO()
    sf.write(buf, full, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/tts")
async def tts(req: TtsRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    voice = (req.voice or DEFAULT_VOICE).strip()
    if not voice:
        raise HTTPException(status_code=400, detail="voice is empty")

    async with _sem:
        try:
            audio_bytes = await asyncio.to_thread(synth_to_wav_bytes, req.text, voice)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return Response(content=audio_bytes, media_type="audio/wav")