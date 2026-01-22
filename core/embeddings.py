from __future__ import annotations

"""Embeddings utilities.

Design goals (per problem statement):
- Use *meaningful* semantic/multimodal vectors when models are available.
- Remain end-to-end runnable offline (no hard crashes) with deterministic fallbacks.

Text:
- Primary: SentenceTransformer (384-d, cosine).
- Fallback: lexical hashing (bag-of-words style) into 384 dims (still meaningful via token overlap).

Image:
- Primary: CLIP ViT-B/32 image features (512-d, cosine).
- Fallback: 8x8x8 RGB color histogram (512-d, cosine) which provides a simple but meaningful visual signature.
"""

import hashlib
import re
from typing import Iterable, List, Union

import numpy as np
from PIL import Image

# -----------------------------
# Text embeddings
# -----------------------------

_TEXT_DIM = 384

_word_re = re.compile(r"[a-z0-9]{2,}")


def _hash_token(token: str) -> int:
    # Stable integer hash for token
    h = hashlib.sha1(token.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _hashing_bow_embed(text: str, dim: int = _TEXT_DIM) -> np.ndarray:
    """Deterministic lexical hashing embedding.

    This is not as strong as a transformer embedding, but it preserves *meaningful similarity*
    via shared tokens and is fully offline.
    """
    v = np.zeros((dim,), dtype=np.float32)
    tokens = _word_re.findall((text or "").lower())
    if not tokens:
        return v

    for t in tokens:
        hv = _hash_token(t)
        idx = hv % dim
        sign = 1.0 if (hv >> 8) & 1 else -1.0
        # simple term-frequency accumulation
        v[idx] += sign

    # sublinear scaling
    v = np.sign(v) * np.sqrt(np.abs(v))
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


_text_model = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        # 384-d model
        _text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _text_model


def embed_text(texts: Union[str, Iterable[str]]) -> np.ndarray:
    """Return normalized 384-d embeddings for text."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    try:
        vecs = _get_text_model().encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)
    except Exception:
        return np.stack([_hashing_bow_embed(t, _TEXT_DIM) for t in texts]).astype(np.float32)


# -----------------------------
# Image embeddings
# -----------------------------

_IMG_DIM = 512

_clip_model = None
_clip_proc = None


def _get_clip():
    global _clip_model, _clip_proc
    if _clip_model is None or _clip_proc is None:
        # Lazy import so the rest of the system can run even if transformers/torch fail.
        import torch
        from transformers import CLIPProcessor, CLIPModel

        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_proc


def _image_hist_512(pil_image: Image.Image) -> np.ndarray:
    """8x8x8 RGB histogram (512-d) fallback embedding."""
    img = pil_image.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    # Compute 3D histogram with 8 bins per channel => 512 dims
    bins = 8
    hist, _ = np.histogramdd(
        arr.reshape(-1, 3),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256)),
    )
    v = hist.astype(np.float32).reshape(-1)
    # Hellinger-style normalization to reduce dominance of large bins
    v = np.sqrt(v)
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def embed_image(pil_image: Image.Image) -> np.ndarray:
    """Return a 512-d image embedding.

    Primary: CLIP image features.
    Fallback: RGB histogram features.
    """
    try:
        import torch

        model, proc = _get_clip()
        inputs = proc(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)[0]
    except Exception:
        return _image_hist_512(pil_image)


def embed_clip_text(texts: Union[str, Iterable[str]]) -> np.ndarray:
    """Return CLIP text embeddings (n,512).

    Primary: CLIP text tower.
    Fallback: lexical hashing into 512 dims.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    try:
        import torch

        model, proc = _get_clip()
        inputs = proc(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)
    except Exception:
        # Hash into 512 dims for consistency with CLIP output size
        vecs = []
        for t in texts:
            v = np.zeros((_IMG_DIM,), dtype=np.float32)
            tokens = _word_re.findall((t or "").lower())
            for tok in tokens:
                hv = _hash_token(tok)
                idx = hv % _IMG_DIM
                sign = 1.0 if (hv >> 8) & 1 else -1.0
                v[idx] += sign
            v = np.sign(v) * np.sqrt(np.abs(v))
            n = float(np.linalg.norm(v) + 1e-12)
            vecs.append((v / n).astype(np.float32))
        return np.stack(vecs).astype(np.float32)


# -----------------------------
# Optional ASR
# -----------------------------


def transcribe_audio(audio_path: str) -> str:
    """Optional audio transcription (offline-friendly).

    Uses faster-whisper if installed; otherwise returns empty string.
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path)
        return " ".join(seg.text.strip() for seg in segments)
    except Exception:
        return ""
