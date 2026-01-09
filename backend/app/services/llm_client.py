import requests

OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "mistral"

def generate_response(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("response", "")
