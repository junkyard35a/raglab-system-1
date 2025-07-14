from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info("✅ rag_server.py loaded")

app = FastAPI()

# Configure paths
DOC_DIR = "data/docs"
DB_DIR = "db/chroma"

# Globals — will be populated during startup
embeddings = None
vectorstore = None

# Supported file extensions
supported_extensions = {
    ".pdf", ".docx", ".xlsx", ".txt", ".py", ".sh", ".tf", ".json", ".yaml", ".yml"
}

### Add post-processing step for better relevance ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def rerank(query, documents):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

### Keyword-based filtering alongside vector search (Hybrid search approach)
class HybridRetriever(BaseRetriever):
    vector_retriever: Chroma
    keyword_field: str = "source"
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Vector search
        vector_docs = self.vector_retriever.similarity_search(query, k=10)
        
        # Keyword search (basic example)
        keyword_results = []
        all_docs = self.vector_retriever.get_all_documents()
        
        for doc in all_docs:
            if query.lower() in doc.page_content.lower():
                keyword_results.append(doc)
        
        # Combine and deduplicate
        combined = {}
        for doc in vector_docs + keyword_results:
            combined[doc.metadata["source"]] = doc
            
        return list(combined.values())[:4]

@app.on_event("startup")
async def startup_event():
    await load_documents()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

### Update retrieve endpoint to handle metadata
@app.get("/retrieve")
async def retrieve(query: str):
    global vectorstore
    if vectorstore is None:
        return {"error": "Vectorstore not initialized"}
    
    try:
        # Get more results initially using similarity search with scores
        raw_results = vectorstore.similarity_search_with_score(query, k=10)
        
        # Deduplicate by source and include scores
        seen_sources = set()
        unique_results = []
        
        for result, score in raw_results:
            source = result.metadata.get("source", "unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_results.append({
                    "content": result.page_content,
                    "source": source,
                    "score": float(score)  # Convert numpy float32 to Python float
                })
                
        # Re-rank final results
        reranked = rerank(query, [Document(page_content=r["content"], metadata={"source": r["source"]}) for r in unique_results])
        
        return {
            "query": query,
            "results": [{
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "score": float(score)
            } for doc, score in reranked[:4]]  # Return top 4 unique candidates
        }
    except Exception as e:
        logging.error(f"Search error: {e}")
        return {"error": "Internal server error"}, 500

### Add an endpoint to force reindexing
@app.post("/reindex")
async def reindex():
    global vectorstore
    try:
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR)
        await load_documents()
        return {"status": "Reindexed successfully"}
    except Exception as e:
        logging.error(f"Reindex error: {e}")
        return {"error": "Reindex failed"}, 500

### Add alternative endpoint for hybrid search approach
@app.get("/hybrid-retrieve")
async def hybrid_retrieve(query: str):
    global vectorstore
    if vectorstore is None:
        return {"error": "Vectorstore not initialized"}
    
    try:
        retriever = HybridRetriever(vector_retriever=vectorstore)
        raw_results = retriever.get_relevant_documents(query)
        reranked = rerank(query, raw_results)
        
        return {
            "query": query,
            "results": [{
                "content": doc.page_content,
                "source": doc.metadata["source"]
            } for doc, _ in reranked]
        }
    except Exception as e:
        logging.error(f"Hybrid search error: {e}")
        return {"error": "Internal server error"}, 500

async def load_documents():
    global embeddings, vectorstore
    logging.info("🔍 Starting document load process...")
    logging.info(f"📁 Current DOC_DIR path: {DOC_DIR}")
    # Add after checking directory existence
    if not os.path.exists(DOC_DIR):
        logging.error(f"❌ DOC_DIR does not exist: {DOC_DIR}")
        return


    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",  # Better for technical concepts
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Important for similarity scores
    )

    # Only reindex if DB doesn't already exist
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        logging.info("📁 Vector DB is empty or missing — proceeding with indexing")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks capture more context
            chunk_overlap=200,  # Better context continuity
            separators=["\n\n", "\n", " ", ""]
        )

        docs = []  # Initialize the docs list
        
        # Loading process to preserve document metadata
        for filename in os.listdir(DOC_DIR):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                path = os.path.join(DOC_DIR, filename)
                try:
                    loader = UnstructuredFileLoader(path)
                    loaded = loader.load()
                    # Add source metadata to each document
                    for doc in loaded:
                        doc.metadata["source"] = filename
                    docs.extend(loaded)
                    logging.info(f"📄 Loaded {len(loaded)} chunks from {filename}")
                except Exception as e:
                    logging.error(f"❌ Error loading {filename}: {e}")

        if not docs:
            logging.warning("⚠️ No documents found — check files and paths")
            return

        splits = text_splitter.split_documents(docs)
        logging.info(f"🧠 Initializing vectorstore and adding {len(splits)} splits...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        logging.info("📚 Vector database initialized with document splits.")
    else:
        logging.info("📂 Vector DB already exists — skipping indexing")
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        logging.info("🧠 Loaded existing vectorstore")
