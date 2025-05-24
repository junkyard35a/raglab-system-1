from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        logging.warning("⚠️ Vectorstore not initialized yet")
        return {"query": query, "results": []}

    try:
        results = vectorstore.similarity_search(query, k=4)
        return {"query": query, "results": [r.page_content for r in results]}
    except Exception as e:
        logging.error(f"Search error: {e}")
        return {"error": "Internal server error"}, 500


async def load_documents():
    global embeddings, vectorstore
    logging.info("🔍 Starting document load process...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Only reindex if DB doesn't already exist
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        logging.info("📁 Vector DB is empty or missing — proceeding with indexing")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        docs = []
        for filename in os.listdir(DOC_DIR):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                path = os.path.join(DOC_DIR, filename)
                try:
                    loader = UnstructuredFileLoader(path)
                    loaded = loader.load()
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
