import os
import tempfile
from fastapi import APIRouter, UploadFile, File
from app.services.ingestion import extract_text_from_pdf, chunk_text

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

    return {
        "num_chunks": len(chunks),
        "chunks": chunks[:5]  # preview only
    }
