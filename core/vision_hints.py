from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from .config import ENABLE_IMAGE_HINTS
from .embeddings import embed_image, embed_clip_text

DEFAULT_LABELS = [
    ("Pothole", "a photo of a pothole on a road"),
    ("Garbage", "a photo of garbage pile on street"),
    ("Streetlight", "a photo of a streetlight at night"),
    ("Water Leak", "a photo of water leak on road"),
    ("Electricity", "a photo of electrical wire or pole"),
    ("Sanitation", "a photo of sewage overflow"),
]

def infer_image_issue_hints(img: Image.Image, labels: List[Tuple[str,str]] = DEFAULT_LABELS, topk: int = 3) -> List[Dict[str, float]]:
    """Returns top issue-category hints from an image using CLIP similarity.
    If CLIP isn't available or disabled, returns [].
    """
    if not ENABLE_IMAGE_HINTS:
        return []
    try:
        ivec = embed_image(img).astype(np.float32)
        if float(np.linalg.norm(ivec)) < 1e-6:
            return []
        texts = [t for _, t in labels]
        tvecs = embed_clip_text(texts)  # (n, 512)
        # cosine similarity (vectors are already normalized in CLIP outputs when available)
        ivec = ivec / (np.linalg.norm(ivec) + 1e-12)
        tvecs = tvecs / (np.linalg.norm(tvecs, axis=1, keepdims=True) + 1e-12)
        sims = (tvecs @ ivec).tolist()
        scored = [{"label": labels[i][0], "score": float(sims[i])} for i in range(len(labels))]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:topk]
    except Exception:
        return []
