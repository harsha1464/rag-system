from fastapi import APIRouter
import requests

router = APIRouter(
    prefix="/llm",
    tags=["LLM"]
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


@router.post("/test")
def test_llm(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return {
        "response": data.get("response", "")
    }
