from fastapi import APIRouter
from pydantic import BaseModel
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_client import generate_response  # you’ll define this soon

router = APIRouter(prefix="/rag", tags=["RAG"])

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/query")
def rag_query(body: RAGRequest):
    embed_svc = EmbeddingService()
    vector_store = VectorStore(dim=embed_svc.model.get_sentence_embedding_dimension())

    # 1. Embed the query
    q_emb = embed_svc.embed_query(body.query)

    # 2. Search FAISS
    hits = vector_store.search(q_emb, body.top_k)

    # 3. Build prompt context
    context_texts = "\n\n".join([h["content"] for h in hits])

    prompt = f"""
    Context:
    {context_texts}

    Question:
    {body.query}

    Answer:
    """

    # 4. Call local LLM
    answer = generate_response(prompt)
    return {
        "answer": answer,
        "retrieved_chunks": hits
    }
