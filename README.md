## 📝 Technical Design Document: RAG System (Personal LAB)

### 1. Overview

**Project Name**: LocalLLM RAG System (Personal Lab).

**Objective**: Build a minimal but functional RAG-based QA system using Qwen-14B and a lightweight backend.  

**Target Audience**: Developers, AI Architects, MLOps Engineers  

---

### 2. Architecture Diagram

- User → Frontend → FastAPI → Chroma (retrieval) → Qwen (generation)

**Infrastructure**

![Alt](images/LocalLLM+RAG-Infrastructure-v11.drawio.png)

---

### 3. Component Breakdown

| Component | Description |
|----------|-------------|
| Document Loader | Uses `PyPDFLoader` to read input files |
| Text Splitter | `RecursiveCharacterTextSplitter` for semantic chunking |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | Chroma for CPU-friendly vector DB |
| Retrieval API | FastAPI endpoint to serve similarity search |
| LLM Generator | Qwen-14B via `llama.cpp` and `OpenWebUI` |
| Query Agent | Combines retrieval + generation |

---

### 4. Data Flow

1. User submits query
2. Query hits `/retrieve` endpoint
3. Vector store returns top-k similar chunks
4. Query agent formats prompt with context
5. LLM generates final answer
6. Response returned to user

---

### 5. Trade-offs & Considerations

| Decision | Why |
|--------|-----|
| Chroma over FAISS | Simpler to set up locally; no extra indexing needed |
| Sentence Transformers | Lightweight and works well with CPU |
| No cloud hosting | Entirely local due to hardware constraints |
| Sync over async API | Simpler implementation for MVP |

---

### 6. Scaling Considerations

- Add caching layer for repeated queries
- Replace Chroma with Weaviate or Pinecone for scalability
- Use Redis for context caching
- Containerize services with Docker
- Kubernetes for orchestration at scale

---

### 7. Work in Progress: Improvements

- Add evaluation metrics (faithfulness, relevance)
- Support more document types (Markdown, Word, etc.)
- Implement logging and monitoring
- Add prompt engineering guardrails
- Explore LoRA fine-tuning for domain-specific answers

---
