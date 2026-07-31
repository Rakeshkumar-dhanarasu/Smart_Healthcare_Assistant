from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hybridrag import RAGService
from dotenv import load_dotenv
import os

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- INIT APP ----------------
app = FastAPI(
    title="Hybrid RAG API",
    description="Graph + Vector RAG Backend",
    version="1.0"
)

# ---------------- CORS CONFIG ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT SERVICE ----------------
rag_service = RAGService()

# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str

# ---------------- ROUTES ----------------

@app.get("/")
def root():
    return {"message": "Hybrid RAG API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        response = rag_service.run_pipeline(request.query)

        return {
            "query": request.query,
            "response": response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- DEBUG ROUTE ----------------
@app.post("/debug")
def debug_rag(request: QueryRequest):
    try:
        debug_data = rag_service.debug_pipeline(request.query)

        return debug_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))