import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral"


def generate_response(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["response"]
