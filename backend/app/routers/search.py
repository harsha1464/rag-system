from fastapi import APIRouter
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/search", tags=["Search"])

embedding_service = EmbeddingService()
vector_store = VectorStore(dim=384)


@router.post("/")
def search(query: str, top_k: int = 5):
    query_embedding = embedding_service.embed_query(query)
    return vector_store.search(query_embedding, top_k)
