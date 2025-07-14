from fastapi import FastAPI
###from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
###from faiss import gpu, StandardGpuResources, IndexFlatIP, IndexIVFPQ
from faiss import StandardGpuResources, IndexFlatIP, IndexIVFPQ
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import logging
import torch
import re  # Missing import for preprocessing
import sentence_transformers
import nltk
import faiss
import numpy as np

logger = logging.getLogger(__name__)

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

dim = 768  # Or dynamically from your embeddings
res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

nltk.data.path.append("/opt/rag_server_venv/nltk_data")  # Or preferred shared path

print("SentenceTransformers version:", sentence_transformers.__version__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info("✅ rag_server.py loaded")

app = FastAPI()

# Configure paths
DOC_DIR = "data/docs"
DB_DIR = "db/faiss"

# Globals — will be populated during startup
embeddings = None
vectorstore = None

# Supported file extensions
supported_extensions = {
    ".pdf", ".docx", ".xlsx", ".txt", ".py", ".sh", ".tf", ".json", ".yaml", ".yml"
}

### Add post-processing step for better relevance ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def rerank(query: str, documents: List[Document], batch_size: int = 64) -> List[tuple]:
    """
    Optimized reranking with batch processing on GPU
    """
    if not documents:
        return []

    # Prepare batches
    doc_contents = [doc.page_content for doc in documents]
    batches = [(query, content) for content in doc_contents]

    # Process in batches to avoid OOM
    all_scores = []
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i + batch_size]
        scores = cross_encoder.predict(batch, show_progress_bar=False)
        all_scores.extend(scores)

    # Combine with documents and sort
    scored_docs = list(zip(documents, all_scores))
    return sorted(scored_docs, key=lambda x: x[1], reverse=True)


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare fields using Pydantic's Field
    vector_db: FAISS = Field(...)
    keyword_weight: float = Field(0.3)
    rerank_top_n: int = Field(50)

    def __init__(self, vector_db: FAISS, **kwargs):
        # Extract HybridRetriever-specific kwargs before calling super()
        keyword_weight = kwargs.pop("keyword_weight", 0.3)
        rerank_top_n = kwargs.pop("rerank_top_n", 50)

        # Initialize parent class with remaining kwargs
        super().__init__(**kwargs)

        # Assign fields (Pydantic handles validation)
        self.vector_db = vector_db
        self.keyword_weight = keyword_weight
        self.rerank_top_n = rerank_top_n

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval logic combining vector and keyword search"""
        try:
            # Get vector results
            vector_docs = self.vector_db.similarity_search(query, k=self.rerank_top_n)
            
            # Get keyword matches
            keyword_matches = self._get_keyword_matches(query)
            
            # Combine and return results
            return self._combine_results(vector_docs, keyword_matches)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise

    def _get_keyword_matches(self, query: str) -> List[tuple[Document, float]]:
        """Keyword matching implementation"""
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
        keyword_matches: List[tuple[Document, float]]
    ) -> List[Document]:
        """Result combination implementation"""
        combined = {}
        
        # Add vector results
        for doc in vector_docs:
            combined[doc.metadata["source"]] = (doc, 1.0)
            
        # Add weighted keyword matches
        for doc, score in keyword_matches:
            source = doc.metadata["source"]
            if source in combined:
                _, existing_score = combined[source]
                combined[source] = (doc, existing_score + (score * self.keyword_weight))
            else:
                combined[source] = (doc, score * self.keyword_weight)
                
        # Return sorted, deduplicated results
        return [doc for doc, _ in sorted(combined.values(), key=lambda x: x[1], reverse=True)[:10]]

@app.on_event("startup")
async def startup_event():
    await load_documents()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/retrieve")
async def retrieve(query: str):
    global vectorstore
    if vectorstore is None:
        return {"error": "Vectorstore not initialized"}

    try:
        raw_results = vectorstore.similarity_search_with_score(query, k=20)

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

        # Move index to GPU after reindexing (it was saved as CPU index)
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
    """Security-focused preprocessing for IAC code"""
    # Remove sensitive data
    content = re.sub(r"password\s*=\s*\"[^\"]+\"", "password = \"REDACTED\"", content)
    
    # Highlight security keywords
    security_keywords = ["iam", "role", "policy", "secret", "key", "security_group"]
    for keyword in security_keywords:
        content = re.sub(
            rf"(\b{keyword}\b)", 
            r"**SECURITY:\1**", 
            content, 
            flags=re.IGNORECASE
        )
    return content

async def load_documents():
    global embeddings, vectorstore
    logging.info("🔍 Starting document load process...")

    # Required for GPU FAISS use in all cases
    res = faiss.StandardGpuResources()

    # === Initialize HuggingFace Embeddings on GPU ===
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={
            "device": "cuda",
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 256,
            "device": "cuda",
        }
    )

    # === Load FAISS Index from Disk if Available ===
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logging.info("📂 Vector DB exists — loading from disk")
        vectorstore = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Move to GPU
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
        return

    # === Load and Preprocess Documents ===
    docs = []
    for filename in os.listdir(DOC_DIR):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_extensions:
            path = os.path.join(DOC_DIR, filename)
            try:
                loader = UnstructuredFileLoader(path)
                loaded = loader.load()
                for doc in loaded:
                    doc.page_content = preprocess_code_chunk(doc.page_content)
                    doc.metadata["source"] = filename
                docs.extend(loaded)
                logging.info(f"📄 Loaded {len(loaded)} chunks from {filename}")
            except Exception as e:
                logging.error(f"❌ Error loading {filename}: {e}")

    if not docs:
        logging.warning("⚠️ No documents found — check DOC_DIR")
        return

    # === Split Documents ===
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=256,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    logging.info(f"🧠 Split into {len(splits)} chunks")

    # === Get Embedding Dim ===
    sample_vector = embeddings.embed_query("dummy")
    dim = len(sample_vector)

    # === Train IVFPQ Index on GPU ===
    training_texts = [doc.page_content for doc in splits[:20000]]
    training_vectors = embeddings.embed_documents(training_texts)
    training_vectors_np = np.array(training_vectors).astype('float32')
    logging.info(f"📊 Training vector shape: {training_vectors_np.shape}")

    MIN_TRAINING_VECTORS = 100
    if training_vectors_np.shape[0] < MIN_TRAINING_VECTORS:
        logging.warning(f"⚠️ Only {training_vectors_np.shape[0]} vectors available — using GPU IndexFlatIP fallback")
        flat_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=gpu_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # Add all documents in one go
        vectorstore.add_documents(splits)

        # Save CPU version for persistence
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        vectorstore.index = cpu_index
        vectorstore.save_local(DB_DIR)
        logging.info("💾 Saved FAISS (FlatIP fallback) index to disk")

        # Reload to GPU for active use
        vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        logging.info("🚀 FlatIP index reloaded to GPU")
        return

    nlist, m, bits = 200, 12, 8
    if training_vectors_np.shape[0] < nlist:
        nlist = training_vectors_np.shape[0]
        logging.warning(f"⚠️ Reduced nlist to {nlist} due to limited training data")

    gpu_config = faiss.GpuIndexIVFPQConfig()
    gpu_config.device = 0

    gpu_index = faiss.GpuIndexIVFPQ(
        res,
        dim,
        nlist,
        m,
        bits,
        faiss.METRIC_INNER_PRODUCT,
        gpu_config
    )

    logging.info("🎯 Training FAISS IVFPQ index on GPU")
    gpu_index.train(training_vectors_np)

    # === Build Vectorstore ===
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=gpu_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    # === Add Document Chunks in Batches ===
    batch_size = 64
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        vectorstore.add_documents(batch)
        print_gpu_memory()

    # === Persist GPU Index (converted to CPU) ===
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    vectorstore.index = cpu_index
    vectorstore.save_local(DB_DIR)
    logging.info("💾 Saved FAISS index to disk")

    # === Reload to GPU for Active Use ===
    vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    logging.info("🚀 FAISS index reloaded to GPU for runtime use")

    # Add logging to confirm the "Department of Education" section is successfully loaded and indexed
    for doc in loaded:
        doc.page_content = preprocess_code_chunk(doc.page_content)
        doc.metadata["source"] = filename
        if "Department of Education" in doc.page_content:
            logging.info(f"📌 Document contains 'Department of Education': {filename}")
    docs.extend(loaded)


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def create_optimized_index(dim: int):
    # Configuration for 8GB GPU
    nlist = 200  # Number of clusters
    m = 12        # Number of subquantizers
    bits = 8     # Bits per subquantizer

    quantizer = faiss.IndexFlatIP(dim)
    gpu_quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)

    # Use IVFPQ for better memory efficiency with large datasets
    index = faiss.IndexIVFPQ(gpu_quantizer, dim, nlist, m, bits)

    # Train with representative sample
    if not index.is_trained:
        logging.info("Training FAISS index...")
        # Generate some random training data
        np.random.seed(42)
        training_data = np.random.rand(10000, dim).astype('float32')
        index.train(training_data)

    return index


@app.get("/system_status")
async def system_status():
    status = {
        "gpu_available": torch.cuda.is_available(),
        "faiss_backend": "GPU" if isinstance(vectorstore.index, faiss.GpuIndex) else "CPU",
        "embedding_device": embeddings.client.device,
        "cross_encoder_device": cross_encoder.device,
    }

    if torch.cuda.is_available():
        status.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
        })

    return status

def create_gpu_ivfpq_index(dim: int, nlist: int = 100, m: int = 8, bits: int = 8) -> faiss.GpuIndexIVFPQ:
    config = faiss.GpuIndexIVFPQConfig()
    config.device = 0  # GPU ID
    config.useFloat16 = True  # optional: reduce memory usage

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFPQ(res, dim, nlist, m, bits, faiss.METRIC_INNER_PRODUCT, config)
    return index

