from __future__ import annotations
from typing import List, Dict, Any

class Reranker:
    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return docs

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Handle empty docs list
        if not docs:
            return docs
        
        pairs = [(query, d.get("text","")) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return docs
