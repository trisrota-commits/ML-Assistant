from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_core import query_rag

app = FastAPI(title="ML RAG Assistant")


# ---- Request Schema ----
class QueryRequest(BaseModel):
    question: str


# ---- Health Check ----
@app.get("/")
def health():
    return {"status": "running"}


# ---- Query Endpoint ----
@app.post("/query")
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, results = query_rag(request.question)

        # Ensure JSON-safe output
        retrieved_ids = []
        if results:
            retrieved_ids = [int(r["id"]) for r in results]

        return {
            "question": request.question,
            "answer": answer,
            "retrieved_ids": retrieved_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))