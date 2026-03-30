from fastapi import FastAPI
from pydantic import BaseModel
import time

from chatbot import api_question_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    college_name: str = "default"

@app.post("/query")
def query_rag(req: QueryRequest):
    start = time.time()

    answer = api_question_answer(req.query, req.college_name)

    end = time.time()

    return {
        "query": req.query,
        "answer": answer,
        "latency": round(end - start, 3)
    }