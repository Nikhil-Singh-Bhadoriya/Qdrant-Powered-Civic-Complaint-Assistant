from __future__ import annotations

"""Pre-download / cache open-source models used by CivicFix.

Why this exists:
- Many evaluation environments block outbound internet.
- Hugging Face models can be cached ahead of time so the full system runs offline.

Usage:
    python scripts/precache_models.py

Optional env vars:
    HF_HOME, TRANSFORMERS_CACHE, SENTENCE_TRANSFORMERS_HOME
    LLM_MODEL (defaults to TinyLlama)
    RERANK_MODEL (defaults to ms-marco MiniLM cross-encoder)
"""

import os

from core.config import LLM_MODEL, RERANK_MODEL

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


def main():
    print("Caching models to local Hugging Face cache...")
    print(f"- Text embedding model: {TEXT_MODEL}")
    print(f"- CLIP model: {CLIP_MODEL}")
    print(f"- Reranker model: {RERANK_MODEL}")
    print(f"- LLM model: {LLM_MODEL}")

    # Text embedder (SentenceTransformers)
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(TEXT_MODEL)
        print("✓ Cached SentenceTransformer")
    except Exception as e:
        print("! Failed caching SentenceTransformer:", e)

    # CLIP (Transformers)
    try:
        from transformers import CLIPModel, CLIPProcessor

        CLIPModel.from_pretrained(CLIP_MODEL)
        CLIPProcessor.from_pretrained(CLIP_MODEL)
        print("✓ Cached CLIP")
    except Exception as e:
        print("! Failed caching CLIP:", e)

    # Reranker (Transformers)
    try:
        from sentence_transformers import CrossEncoder

        CrossEncoder(RERANK_MODEL)
        print("✓ Cached CrossEncoder reranker")
    except Exception as e:
        print("! Failed caching reranker:", e)

    # Local LLM (Transformers)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
        AutoModelForCausalLM.from_pretrained(LLM_MODEL, low_cpu_mem_usage=True)
        print("✓ Cached Local LLM")
    except Exception as e:
        print("! Failed caching Local LLM:", e)

    print("Done.")


if __name__ == "__main__":
    main()
