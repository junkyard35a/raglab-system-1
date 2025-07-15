## 📝 Technical Design Document: RAG System (Personal LAB) - Version 3

### 1. Overview

**Project Name**: LocalLLM RAG System (Personal Lab).

**Objective**: Build a minimal but functional RAG-based QA system with Qwen3-14B as a backend LLM  

**Target Audience**: Developers, AI Architects, MLOps Engineers  

**Value proposition**: [Use Cases](USECASES.md) with 100% context and data isolation.

---

### 2. Architecture

#### 2.1 TL;DR
- User → Frontend → FastAPI → RAG → Qwen (LLM)

#### 2.2 Infrastructure

![LLM+RAG-Infra-diagram](artefacts/images/LocalLLM+RAG-Infrastructure-v30.drawio.png)

#### 2.3 Hardware Specs
| Host | Function | GPU | Hardware | OS | LLM |
|------|----------|-----|----------|----|-----|
| ds1 | Local LLM | Geforce RTX 4060 16GB VRAM | AMD Ryzen 5 5500, Gigabyte B550M DS3H AC, Lexar SSD NM790 2TB, 64GB RAM | Ubuntu2404LTS | Qwen3-14b Q6_K_XL |
| rag1 | UI + RAG | Geforce GTX 1070 8GB VRAM | AMD Ryzen 5 5500, Asus PRIME B550M-K, Kingston SNV2S1000G 1TB, 32GB RAM | Ubuntu2404LTS | Qwen3-0.6b embedding BF16 |

---

### 3. Component Breakdown

| Service | Function | Description |
|---------|----------|-------------|
| rag_server | Document Loader | Uses `UnstructuredFileLoader` to read input files |
| rag_server | Text Splitter | `RecursiveCharacterTextSplitter` for semantic chunking |
| rag_server | Embedding Model | `Qwen3-0.6b embedding BF16` |
| rag_server | Vector Store | FAISS for GPU based vector DB |
| rag_proxy | Retrieval API | FastAPI endpoint to serve similarity search |
| rag_proxy | Query Agent | Combines retrieval + generation |
| localLLM | LLM | Qwen-14B Q6_K_XL via llama.cpp/llama-server |
| openwebui | Webfront chat | OpenWebUI |

---

### 4. Data Flow

1. User submits query via webfront chat
2. Query hits `/retrieve` endpoint
3. Vector store returns top-k similar chunks
4. Query agent formats prompt with context
5. LLM generates final answer
6. Response returned to user

---

### 5. Trade-offs & Considerations

| Decision | Why |
|--------|-----|
| FAISS | GPU acceleration support and embedding management separated from metadata |
| Sentence Transformers | Lightweight |
| No cloud hosting | Entirely local for security and privacy use cases |
| Sync over async API | Simpler implementation for MVP |

---

### 6. Next Steps

#### 6.1 Scaling Considerations
- Add caching layer for repeated queries
- Use Redis for context caching
- Kubernetes for orchestration at scale

#### 6.2 Improvements in progress
- Add evaluation metrics (faithfulness, relevance)
- Implement logging and monitoring
- Add prompt engineering guardrails
- Explore LoRA fine-tuning for domain-specific answers

#### 6.3 Fixes in progress
- Fix streaming tokenization issues between LLM through rag proxy to webfront
- Fix rag proxy and openwebui non-streaming think appearing in answers
---
