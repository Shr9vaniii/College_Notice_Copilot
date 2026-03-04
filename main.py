import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from ingestion import get_college_retriever,extract_text_from_pdf,add_text
from chromadb import CloudClient
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from ingestion import get_text_processed
from google import genai
from ingestion import client,extract_text_from_pdf



load_dotenv(override=True)

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

def main():
    print("Hello from college-chatbot!")
    pdf_path="insem_tt.pdf"
    extract_text_from_pdf(pdf_path,'pict',client)
    

      
    
   


    

if __name__ == "__main__":
    main()
