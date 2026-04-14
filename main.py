from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form,Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
import asyncio
import os
from pathlib import Path
from chatbot import question_answer
from typing import List, Dict
from ingestion import add_text,process_pdfs
from fastapi import Depends
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from firebase_admin import firestore
import boto3
import tempfile
from dotenv import load_dotenv
from cache import r
from contextlib import asynccontextmanager

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION')
)

s3_bucket_name=os.getenv('S3_BUCKET_NAME')

cred_path=os.getenv('FIREBASE_CREDENTIALS')
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

db=firestore.client()

app = FastAPI()
security = HTTPBearer()

async def verify_user(request:Request):
    auth_header=request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("errorrrrrrrr")
        raise HTTPException(status_code=401, detail="Unauthorized")
    token=auth_header.split("Bearer ")[1]
    try:
        decoded_token=auth.verify_id_token(token)
        user_id=decoded_token["uid"]
        user_doc=db.collection("users").document(user_id).get()
        if not user_doc.exists:
            print("errorrrrrrrr")
            raise HTTPException(status_code=401, detail="Unauthorized")

        user_data=user_doc.to_dict()
        return user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev (IMPORTANT)
    allow_credentials=True,
    allow_methods=["*"],  # allows OPTIONS, POST, etc.
    allow_headers=["*"],
)

limiter=Limiter(key_func=get_remote_address)
app.state.limiter=limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class QueryRequest(BaseModel):
    question: str
    history:List[Dict]=[]

@app.post("/student/chat")
#@limiter.limit("20/second")
async def query_rag(request:Request,query: QueryRequest,token: str = Depends(security)):
    user=None
    
    start = time.time() 
    try:
        user=await verify_user(request)
        if user["role"] != "student":
            raise HTTPException(status_code=403, detail="Admins only")

        
    except Exception as e:
        raise HTTPException(status_code=401, detail="Unauthorized")    


    for attempt in range(2):
        try:
            answer = await question_answer(query.question, query.history,college_name=user["college"])
            end = time.time()

            return {
                
                "answer": answer[-1]["content"],
                "latency": round(end - start, 3)
            }

        except asyncio.CancelledError:
            print("Request cancelled")
            return {"answer": "Request cancelled, please try again"}

        except Exception as e:
            print("Error:", e)

            if "rate limit" in str(e).lower():
                if attempt < 1:  # retry only once
                    await asyncio.sleep(1)
                    continue  # 🔥 THIS IS KEY
                else:
                    return {
                        "query": query.query,
                        "answer": "⚠️ LLM busy, try again",
                        "latency": round(time.time() - start, 3)
                    }

            else:
                raise e



@app.post("/admin/upload")
async def upload_file(
    request:Request,
    file: UploadFile = File(None),
    text: str = Form(None),
):
    
    try:
        user=await verify_user(request)
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admins only")

        college_name = user["college"]

        if file:

            s3_key=f"{college_name}/{file.filename}"
            s3.upload_fileobj(file.file, s3_bucket_name, s3_key)
            await process_pdfs(s3_key,college_name)
            return {"message": "Upload successful", "key": s3_key}


        if text:
            await asyncio.to_thread(
                add_text,
                text,
                college_name,
                
            )
            return {"message": "Text added successfully"}

        raise HTTPException(status_code=400, detail="No input provided")

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Upload failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Redis ping:", r.ping())
    except Exception as e:
        print("Redis connection failed:", e)

    print("App started")

    yield

    print("App shutting down")
