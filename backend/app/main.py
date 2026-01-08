from fastapi import FastAPI
from app.routers import health, llm

app = FastAPI(
    title="RAG System Backend",
    description="FastAPI backend with local LLM integration",
    version="0.1.0"
)

# Register routers
app.include_router(health.router)
app.include_router(llm.router)
