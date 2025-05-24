from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import uvicorn
import json
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Your LLM endpoint (must support OpenAI API)
QWEN_COMPLETION_URL = "http://172.17.0.1:10000/v1/chat/completions"
RAG_SERVER = "http://rag-server:18000/retrieve"


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


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    messages = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages in request"}, status_code=400)

    # Retrieve RAG context
    user_query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
    context = ""
    if user_query:
        try:
            rag_response = requests.get(f"{RAG_SERVER}?query={user_query}")
            if rag_response.status_code == 200:
                rag_context = "\n\n".join(rag_response.json().get("results", []))
                # Prepend context as system message
                messages = [{"role": "system", "content": f"Use the following context:\n{rag_context}"}] + messages
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
