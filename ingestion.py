import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core import document_loaders
import pdfplumber
from langchain_core.documents import Document
import chromadb
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

load_dotenv(override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
llm=OpenAI(api_key=gemini_api_key,base_url=gemini_url)

chroma_api=os.getenv('CHROMA_API_KEY')

client = chromadb.CloudClient(
  api_key=chroma_api,
  tenant='246553cc-048a-47fa-aa2a-cf9d61280656',
  database='Project'
)

def add_text(text,college_id,client):
    docs=[]
    college_collection = f"college_{college_id}"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if text:
        docs.append(Document(page_content=text, metadata={"source": college_id, "college_id": college_collection}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    
    vectorestore = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client, collection_name=college_collection)    
    
    print(f"Vector store created with {vectorestore._collection.count()} documents")
    return 


def extract_text_from_pdf(pdf_path, college_id, client):
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

            
    if text:
        text_processed=get_text_processed(text)
        docs.append(Document(page_content=text_processed, metadata={"source": pdf_path, "college_id": college_collection}))   
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    
    vectorestore = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client, collection_name=college_collection)    
    
    print(f"Vector store created with {vectorestore._collection.count()} documents")
    return 
               


def get_college_retriever(college_id, client, k=10):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        client=client,
        collection_name=f"college_{college_id}",
        embedding_function=embeddings
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
        response = llm.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return response.choices[0].message.content   




