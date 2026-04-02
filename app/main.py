from fastapi import FastAPI
from pydantic import BaseModel
from src.generate import generate_answer

app = FastAPI(title="Production RAG API")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Production RAG API running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    answer, docs = generate_answer(request.query)

    sources = []
    for i, doc in enumerate(docs, 1):
        sources.append({
            "id": i,
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return {
        "query": request.query,
        "answer": answer,
        "sources": sources
    }