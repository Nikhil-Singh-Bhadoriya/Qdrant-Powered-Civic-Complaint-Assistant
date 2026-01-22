# Qdrant Convolve Hackathon – Civic AI Agent

## 1. Problem Statement
Urban citizens face difficulties in filing, tracking, and escalating civic complaints due to fragmented portals, unclear jurisdiction mapping, and lack of transparency. This project addresses **Civic Tech & Governance** by building an AI-powered civic assistant that helps users identify the correct department, channel, and procedure for grievance redressal.

## 2. System Design
The system is built as a modular AI agent with Qdrant as the core retrieval and memory engine. It follows a retrieval-first architecture where decisions are driven by vector search results rather than pure generation.

**Key components:**
- Client layer (Web UI + API)
- FastAPI gateway with validation and rate limiting
- Agent orchestrator (intent detection, slot filling, tool calls)
- Qdrant vector database (primary core)
- Optional open-source LLM for response phrasing
- Feedback loop for memory reinforcement

Qdrant is critical because it enables scalable hybrid search, metadata filtering, multimodal vectors, and long-term memory updates.

## 3. Multimodal Strategy
**Data types used:**
- Text: civic procedures, templates, jurisdiction rules
- Images: complaint photos / screenshots (optional)
- Metadata: city, ward, department, category, language

**Embeddings:**
- Text embeddings via open-source sentence transformers
- Image embeddings via CLIP-style models

All embeddings are stored and queried using Qdrant collections with payload filters.

## 4. Search, Memory & Recommendation Logic
- **Search:** Hybrid dense + metadata filtering over multiple Qdrant collections
- **Memory:** 
  - Short-term session memory (TTL-based)
  - Long-term user memory (Qdrant-backed with update/delete/decay)
- **Recommendations:** Ranked channels and departments based on jurisdiction match, urgency, availability, and user preferences

Outputs include evidence snippets, similarity scores, and traceable reasoning paths.

## 5. Limitations & Ethics
**Limitations:**
- OCR and speech-to-text are optional and may be disabled
- Jurisdiction data accuracy depends on source freshness

**Ethics & Safety:**
- PII redaction enabled by default
- No hallucinated responses (retrieval-first design)
- Transparent recommendations with evidence
- Open-source models only; no proprietary APIs

## 6. Conclusion
This project demonstrates a production-aware, responsible AI system using Qdrant for societal impact. It fulfills all mandatory technical requirements of the Convolve 4.0 hackathon while emphasizing transparency, explainability, and reproducibility.
