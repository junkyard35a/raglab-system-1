from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Any, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from faiss import StandardGpuResources, IndexFlatIP
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import logging
import torch
import re
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Paths
DOC_DIR = "data/docs"
DB_DIR = "db/faiss"

# Globals — will be populated during startup
embeddings = None
vectorstore = None

# Supported file extensions
supported_extensions = {
    ".pdf", ".docx", ".xlsx", ".txt", ".py", ".sh", ".tf", ".json", ".yaml", ".yml", ".xaml", ".cs"
}

# Cross encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def rerank(query: str, documents: List[Document], batch_size: int = 16) -> List[tuple]:
    """Optimized reranking with batch processing on GPU"""
    if not documents:
        return []
    doc_contents = [doc.page_content for doc in documents]
    batches = [(query, content) for content in doc_contents]
    all_scores = []
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i + batch_size]
        scores = cross_encoder.predict(batch, show_progress_bar=False)
        all_scores.extend(scores)
    scored_docs = list(zip(documents, all_scores))
    return sorted(scored_docs, key=lambda x: x[1], reverse=True)

def get_available_folders():
    folders = set()
    for root, _, files in os.walk(DOC_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                rel_dir = os.path.relpath(root, DOC_DIR)
                folders.add(rel_dir)
    return sorted(folders)


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_db: FAISS = Field(...)
    keyword_weight: float = Field(0.3)
    rerank_top_n: int = Field(50)

    def __init__(self, vector_db: FAISS, **kwargs):
        keyword_weight = kwargs.pop("keyword_weight", 0.3)
        rerank_top_n = kwargs.pop("rerank_top_n", 50)
        super().__init__(**kwargs)
        self.vector_db = vector_db
        self.keyword_weight = keyword_weight
        self.rerank_top_n = rerank_top_n

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            vector_docs = self.vector_db.similarity_search(query, k=self.rerank_top_n)
            keyword_matches = self._get_keyword_matches(query)
            return self._combine_results(vector_docs, keyword_matches)
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise

    def _get_keyword_matches(self, query: str) -> List[Tuple[Document, float]]:
        query_words = set(re.findall(r'\w+', query.lower()))
        matches = []
        for doc_id in self.vector_db.index_to_docstore_id.values():
            doc = self.vector_db.docstore.search(doc_id)
            if doc:
                doc_words = set(re.findall(r'\w+', doc.page_content.lower()))
                match_score = len(query_words & doc_words) / len(query_words)
                if match_score > 0.2:
                    matches.append((doc, match_score))
        return matches

    def _combine_results(
        self,
        vector_docs: List[Document],
        keyword_matches: List[Tuple[Document, float]]
    ) -> List[Document]:
        combined = {}
        for doc in vector_docs:
            combined[doc.metadata["source"]] = (doc, 1.0)
        for doc, score in keyword_matches:
            source = doc.metadata["source"]
            if source in combined:
                _, existing_score = combined[source]
                combined[source] = (doc, existing_score + (score * self.keyword_weight))
            else:
                combined[source] = (doc, score * self.keyword_weight)
        return [doc for doc, _ in sorted(combined.values(), key=lambda x: x[1], reverse=True)[:10]]

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await load_documents()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/retrieve")
async def retrieve(query: str, folder: Optional[str] = None):
    global vectorstore
    if vectorstore is None:
        return {"error": "Vectorstore not initialized"}
    
    try:
        kwargs = {}
        if folder:
            kwargs["filter"] = {"folder": folder}

        raw_results = vectorstore.similarity_search_with_score(query, k=20, **kwargs)
        seen_sources = set()
        unique_results = []
        for result, score in raw_results:
            source = result.metadata.get("source", "unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_results.append({
                    "content": result.page_content,
                    "source": source,
                    "score": float(score),
                    "metadata": result.metadata
                })
        
        reranked = rerank(
            query,
            [Document(page_content=r["content"], metadata=r["metadata"]) for r in unique_results]
        )
        
        return {
            "query": query,
            "results": [{
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "score": float(score),
                "metadata": doc.metadata
            } for doc, score in reranked[:4]]
        }
    except Exception as e:
        logging.error(f"❌ Search error: {e}")
        return {"error": "Internal server error"}, 500

@app.post("/reindex")
async def reindex():
    global vectorstore
    try:
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR)
        await load_documents()
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
        return {"status": "Reindexed and moved to GPU successfully"}
    except Exception as e:
        logging.error(f"❌ Reindex error: {e}")
        return {"error": "Reindex failed"}, 500

@app.get("/hybrid-retrieve")
async def hybrid_retrieve(query: str):
    if vectorstore is None:
        return {"error": "Vectorstore not initialized"}
    try:
        retriever = HybridRetriever(
            vector_db=vectorstore,
            keyword_weight=0.5,
            rerank_top_n=100
        )
        results = retriever.get_relevant_documents(query)
        reranked = rerank(query, results)
        return {
            "query": query,
            "results": [{
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "metadata": doc.metadata
            } for doc, _ in reranked]
        }
    except Exception as e:
        logging.error(f"Hybrid search error: {e}")
        return {"error": str(e)}, 500

@app.get("/index/status")
async def index_status():
    return {
        "index_type": type(vectorstore.index).__name__,
        "is_trained": vectorstore.index.is_trained,
        "ntotal": vectorstore.index.ntotal
    }

@app.get("/gpu/status")
async def gpu_status():
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0)/1e9:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0)/1e9:.2f}GB"
        }
    return {"status": "No GPU available"}

def preprocess_code_chunk(content: str) -> str:
    content = re.sub(r"password\s*=\s*\"[^\"]+\"", "password = \"REDACTED\"", content)
    security_keywords = ["iam", "role", "policy", "secret", "key", "security_group"]
    for keyword in security_keywords:
        content = re.sub(
            rf"(\b{keyword}\b)",
            r"**SECURITY:\1**",
            content,
            flags=re.IGNORECASE
        )
    return content

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

class Qwen3EmbeddingFunction:
    def __init__(self, max_length=16384, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
        self.model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", torch_dtype=torch.float16).to(device or "cuda")
        self.device = device or "cuda"
        self.max_length = max_length

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        task = "Given a web search query, retrieve relevant passages that answer the query"
        prompt_query = get_detailed_instruct(task, query)
        return self.embed([prompt_query])[0]

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self.embed(docs)


class CustomEmbeddings:
    def __init__(self):
        self.client = Qwen3EmbeddingFunction()
        self.dim = len(self.client.embed(["dummy"])[0])

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    def __call__(self, input: Union[str, List[str]]) -> List[float]:
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise TypeError(f"Input must be string or list of strings. Got: {type(input)}")

async def load_documents():
    global embeddings, vectorstore
    logging.info("🔍 Starting document load process...")
    res = faiss.StandardGpuResources()

    # Use the custom embedding class
    embeddings = CustomEmbeddings()

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logging.info("📂 Vector DB exists — loading from disk")
        vectorstore = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        if torch.cuda.is_available():
            vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
        return

    docs = []

    # Recursively walk through DOC_DIR
    for root, _, files in os.walk(DOC_DIR):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                path = os.path.join(root, filename)
                try:
                    loader = UnstructuredFileLoader(path)
                    loaded = loader.load()
                    for doc in loaded:
                        doc.page_content = preprocess_code_chunk(doc.page_content)
                        doc.metadata["source"] = filename
                        doc.metadata["file_path"] = path  # full system path
                        relative_path = os.path.relpath(path, DOC_DIR)
                        doc.metadata["folder"] = os.path.dirname(relative_path)
                    docs.extend(loaded)
                    logging.info(f"📄 Loaded {len(loaded)} chunks from {path}")
                except Exception as e:
                    logging.error(f"❌ Error loading {path}: {e}")

    if not docs:
        logging.warning("⚠️ No documents found — check DOC_DIR")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=256,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    logging.info(f"🧠 Split into {len(splits)} chunks")

    dim = embeddings.dim

    training_texts = [doc.page_content for doc in splits[:20000]]
    if len(training_texts) < 100:
        logging.warning("⚠️ Not enough data to train IVFPQ index")
        flat_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=gpu_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vectorstore.add_documents(splits)
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        vectorstore.index = cpu_index
        vectorstore.save_local(DB_DIR)
        vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        logging.info("🚀 FlatIP index loaded to GPU")
        return

    nlist, m, bits = 200, 12, 8
    if len(training_texts) < nlist:
        nlist = len(training_texts)

    gpu_config = faiss.GpuIndexIVFPQConfig()
    gpu_config.device = 0
    gpu_config.useFloat16 = True

    training_vectors = embeddings.embed_documents(training_texts)
    training_vectors_np = np.array(training_vectors).astype('float32')

    gpu_index = faiss.GpuIndexIVFPQ(
        res, dim, nlist, m, bits, faiss.METRIC_INNER_PRODUCT, gpu_config
    )
    logging.info("🎯 Training FAISS IVFPQ index on GPU")
    gpu_index.train(training_vectors_np)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=gpu_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    batch_size = 64
    for i in range(0, len(splits), batch_size):
        vectorstore.add_documents(splits[i:i + batch_size])
        print_gpu_memory()

    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    vectorstore.index = cpu_index
    vectorstore.save_local(DB_DIR)
    logging.info("💾 Saved FAISS index to disk")

    vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    logging.info("🚀 FAISS index reloaded to GPU")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

@app.get("/system_status")
async def system_status():
    status = {
        "gpu_available": torch.cuda.is_available(),
        "faiss_backend": "GPU" if isinstance(vectorstore.index, faiss.GpuIndex) else "CPU",
        "embedding_dim": embeddings.dim if embeddings else None,
    }
    if torch.cuda.is_available():
        status.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0)/1e9:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0)/1e9:.2f}GB"
        })
    return status

@app.get("/folders")
async def list_folders():
    return {"folders": get_available_folders()}
