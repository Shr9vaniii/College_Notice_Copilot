import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_core import document_loaders
import pdfplumber
from langchain_core.documents import Document
import chromadb
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from groq import Groq
import datetime as dt
import time
import os
import tempfile
import boto3
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION')
)

s3_bucket_name=os.getenv('S3_BUCKET_NAME')

client1 = Groq()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash,temperature=0")

chroma_api=os.getenv('CHROMA_API_KEY')


client = chromadb.CloudClient(
        api_key=chroma_api,
        tenant="246553cc-048a-47fa-aa2a-cf9d61280656",
        database="Project"
    )


embedding_model=None
retriever_cache={}

def add_text(text,college_id,client=client):
    docs=[]
    college_collection = f"college_{college_id}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    upload_timestamp = dt.datetime.now()
    upload_timestamp_str = upload_timestamp.strftime('%Y-%m-%d %H:%M')
    if text:
        docs.append(Document(page_content=text, metadata={"source": college_id, "college_id": college_collection, "upload_date": upload_timestamp_str}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    
    vectorestore = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client, collection_name=college_collection)    
    
    print(f"Vector store created with {vectorestore._collection.count()} documents")
    return 


def extract_text_from_pdf(pdf_path, college_id, client=client):
    start=time.time()
    docs = []
    text = ""
    college_collection = f"college_{college_id}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore1 = Chroma(
        client=client,
        collection_name=college_collection,
        embedding_function=embeddings
    )
    exists = vectorstore1._collection.get(
        where={"source": pdf_path},
        limit=1
    )

    if exists["ids"]:
        print(f"'{pdf_path}' already exists in college_{college_id}")
        return
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, pages in enumerate(pdf.pages):
            text += pages.extract_text()
    if len(text.strip()) < 50:
        print("Falling back to ocr...")
        images=convert_from_path(pdf_path,dpi=300)
        ocr_text=[]
        for image in images:
            ocr_text.append(pytesseract.image_to_string(image,lang='eng'))    
        text="\n".join(ocr_text)

    upload_timestamp = dt.datetime.now()
    upload_timestamp_str = upload_timestamp.strftime('%Y-%m-%d %H:%M')

    if text:
        text_processed=get_text_processed(text)
        docs.append(Document(page_content=text_processed,
         metadata=
         {"source": pdf_path, 
         "college_id": college_collection,
         "upload_date": upload_timestamp_str}
        )
         )   
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    
    vectorestore = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client, collection_name=college_collection)    
    
    print(f"Vector store created with {vectorestore._collection.count()} documents")
    print(f"Time taken: {time.time() - start} seconds")
    return 
               


def get_college_retriever(college_id, client, k=10):
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    if college_id in retriever_cache:
        return retriever_cache[college_id]    

    vectorstore = Chroma(
        client=client,
        collection_name=f"college_{college_id}",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever
   
def get_text_processed(text):
        system_msg=f'''You are an excellent text_processor. You will be provided with some unprocessed text which can be text extracted from tables,images, etc. Process the text so it can be easily chunked into meaningful data to be stored in vector database.
        Response only the processed text.
        '''
        user_msg=f'''Process the following text.
        The text is:
        {text}
        '''
        response = llm.invoke(messages=[SystemMessage(content=system_msg),HumanMessage(content=user_msg)])
        return response.content   

async def process_pdfs(s3_key,college_id):
    print(f"Processing PDF: {s3_key}")
    if not s3_bucket_name :
        raise ValueError("Missing bucket")

    if not s3_key:
        raise ValueError("Missing key")

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            s3.download_fileobj(s3_bucket_name, s3_key, tmp)
            temp_path = tmp.name
            print(f"📥 Downloaded to temp: {temp_path}")

            # 2. Run your existing ingestion pipeline
            # We use to_thread because PDF processing is CPU-heavy and blocking
            await asyncio.to_thread(
                extract_text_from_pdf, 
                temp_path, 
                college_id
            )
            
            print(f"✅ Ingestion complete for {college_id}")

    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
        raise e

    finally:
            # 3. Always clean up the local server storage
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("🧹 Local temp file removed.")

    return True


