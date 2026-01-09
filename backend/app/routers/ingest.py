import os
import tempfile
from fastapi import APIRouter, UploadFile, File
from app.services.ingestion import extract_text_from_pdf, chunk_text
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

embedding_service = EmbeddingService()
vector_store = VectorStore(dim=384)

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"]
)

@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(text)

    os.remove(tmp_path)

    embeddings = embedding_service.embed_texts(chunks)
    vector_store.add(embeddings, chunks)

    return {
        "num_chunks": len(chunks),
        "chunks": chunks[:5]  # preview only
    }
