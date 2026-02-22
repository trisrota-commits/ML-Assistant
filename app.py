from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_core import query_rag

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    answer, retrieved = query_rag(request.question)
    return {
        "answer": answer,
        "retrieved_chunks": retrieved
    }