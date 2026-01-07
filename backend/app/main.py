from fastapi import FastAPI
from app.llm import generate_response

app = FastAPI(
    title="RAG System Backend",
    description="Backend service for a the RAG system",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-backend"}

@app.post("/llm/test")
def test_llm(prompt: str):
    result = generate_response(prompt)
    return {"response": result}
