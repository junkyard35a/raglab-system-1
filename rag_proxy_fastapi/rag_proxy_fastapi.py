from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import uvicorn
import json
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration from environment variables
QWEN_COMPLETION_URL = os.getenv("QWEN_COMPLETION_URL", "http://172.17.0.1:10000/v1/chat/completions")
RAG_SERVER_BASE = os.getenv("RAG_SERVER_URL", "http://rag-server:18000")
RAG_RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "default")  # default or hybrid

# Construct RAG server URLs
RAG_RETRIEVE_URL = f"{RAG_SERVER_BASE}/retrieve"
RAG_HYBRID_URL = f"{RAG_SERVER_BASE}/hybrid-retrieve"
RAG_REINDEX_URL = f"{RAG_SERVER_BASE}/reindex"

def generate_openai_stream(messages):
    """Stream response from LLM with full conversation history"""
    payload = {
        "model": "qwen3",
        "messages": messages,
        "max_tokens": 20000,
        "stream": True
    }

    try:
        with requests.post(QWEN_COMPLETION_URL, json=payload, stream=True) as resp:
            if resp.status_code != 200:
                yield 'data: {"error": "Model server unreachable"}\n\n'
                return

            for line in resp.iter_lines():
                if not line:
                    continue

                raw_line = line.decode("utf-8").strip()
                logger.info(f"Raw model stream line: {raw_line}")

                if raw_line.startswith("data:"):
                    raw_line = raw_line[5:].strip()

                content = ""
                try:
                    data_json = json.loads(raw_line)
                    content = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "").strip()
                except (json.JSONDecodeError, IndexError, KeyError):
                    content = raw_line.strip()

                if content in ["", "[DONE]"]:
                    continue

                # Build OpenAI-style chunk
                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "qwen-rag-proxy:latest",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }
                    ]
                }

                yield f"data: {json.dumps(chunk)}\n\n"

            # Final stop chunk
            final_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen-rag-proxy:latest",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }

            yield f"data: {json.dumps(final_chunk)}\n\n"

    except Exception as e:
        logger.error(f"Stream generation failed: {str(e)}", exc_info=True)
        error_chunk = {
            "error": str(e),
            "model": "qwen-rag-proxy:latest"
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


def get_full_completion(messages):
    """Get full completion from LLM with conversation history"""
    payload = {
        "model": "qwen3",
        "messages": messages,
        "max_tokens": 20000,
        "stream": False
    }

    try:
        with requests.post(QWEN_COMPLETION_URL, json=payload) as resp:
            if resp.status_code != 200:
                return {"error": "Model server unreachable"}

            data = resp.json()
            full_content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen-rag-proxy:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", -1),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", -1),
                    "total_tokens": data.get("usage", {}).get("total_tokens", -1)
                }
            }

    except Exception as e:
        logger.error(f"Full completion failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def format_rag_context(results):
    """Format retrieved documents with metadata for LLM consumption"""
    if not results:
        return ""
    
    formatted_docs = []
    for i, doc in enumerate(results):
        source = doc.get("source", "unknown")
        score = doc.get("score", 0)
        content = doc.get("content", "")
        
        # Format document with metadata
        formatted_doc = (
            f"[Document {i+1}]\n"
            f"Source: {source}\n"
            f"Relevance Score: {score:.2f}\n"
            f"Content: {content}\n"
        )
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    messages = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages in request"}, status_code=400)

    # Get query from last user message
    user_query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
    context = ""
    
    # Add RAG context if available
    if user_query:
        try:
            # Choose retrieval method based on configuration
            retrieval_url = RAG_HYBRID_URL if RAG_RETRIEVAL_MODE == "hybrid" else RAG_RETRIEVE_URL
            
            # Add query parameters
            params = {
                "query": user_query,
                "use_rag": data.get("use_rag", True)  # Allow override via request param
            }
            
            rag_response = requests.get(retrieval_url, params=params)
            
            if rag_response.status_code == 200:
                results = rag_response.json().get("results", [])
                rag_context = format_rag_context(results)
                
                if rag_context:
                    # Prepend context as system message
                    system_prompt = (
                        "You are a helpful assistant that uses provided context to answer questions. "
                        "Please cite your sources when referencing information from the context. "
                        "Context:\n" + rag_context
                    )
                    messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                logger.warning(f"RAG server returned status code {rag_response.status_code}")
                
        except Exception as e:
            logger.warning(f"RAG server unreachable: {str(e)}")

    # Forward to LLM
    if data.get("stream", False):
        return StreamingResponse(
            generate_openai_stream(messages),
            media_type="text/event-stream"
        )
    else:
        return JSONResponse(get_full_completion(messages))


@app.get("/v1/models")
def v1_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-rag-proxy:latest",
                "object": "model",
                "created": 1719416520,
                "owned_by": "rag-proxy"
            }
        ]
    }


@app.get("/api/version")
def api_version():
    return {"version": "4.3.0"}


@app.get("/api/tags")
def api_tags():
    return {
        "models": [
            {
                "name": "qwen-rag-proxy:latest",
                "model": "qwen-rag-proxy:latest",
                "modified_at": "2025-05-21T00:00:00Z"
            }
        ]
    }


@app.post("/reindex")
def trigger_reindex():
    """Endpoint to trigger reindexing on the RAG server"""
    try:
        response = requests.post(RAG_REINDEX_URL)
        return JSONResponse(
            content={"status": "success", "message": "Reindexing triggered"},
            status_code=response.status_code
        )
    except Exception as e:
        logger.error(f"Reindex trigger failed: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
