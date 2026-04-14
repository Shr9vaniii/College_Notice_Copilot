import os
from dotenv import load_dotenv
from chromadb import HttpClient,CloudClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ingestion import get_college_retriever
from pydantic import BaseModel,Field
from openai import OpenAI
from sentence_transformers import CrossEncoder
import time
from groq import Groq
from chromadb.config import Settings



# load once globally
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv(override=True)

client1 = Groq()

chroma_api=os.getenv('CHROMA_API_KEY')


client = CloudClient(
        api_key=chroma_api,
        tenant="246553cc-048a-47fa-aa2a-cf9d61280656",
        database="Project"
    )
    # This line will confirm if the connection is ACTUALLY alive



MODEL = "gemini-2.5-flash"

llm=ChatGoogleGenerativeAI(model=MODEL,temperature=0)

AVERAGE_CHUNK_SIZE = 500
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))


FINAL_K=5



class Result(BaseModel):
    page_content: str
    metadata: dict   

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )     

class CollegeChatbot:
    def __init__(self):
        self.client = client
        self.llm = llm
        self.llm2=client1
        
       
        self.AVERAGE_CHUNK_SIZE = AVERAGE_CHUNK_SIZE
    def gradio_history_to_messages(self, history):
        messages = []
        for his in history:
            if his['role']=='user':
                messages.append(HumanMessage(content=his['content']))
            elif his['role']=='assistant':
                messages.append(AIMessage(content=his['content']))    
        return messages 

    def merge_chunks(self,chunks, reranked):
        merged = chunks[:]
        existing = [chunk.page_content for chunk in chunks]
        for chunk in reranked:
            if chunk.page_content not in existing:
                merged.append(chunk)
        return merged

 
    def rerank(self, question, chunks):
        # prepare (query, chunk) pairs
        pairs = [(question, chunk.page_content) for chunk in chunks]

        # get scores
        scores = reranker_model.predict(pairs)
        ranked_by_semantic=list(zip(chunks, scores))

        # sort by score descending
        ranked = sorted(
            ranked_by_semantic,
            key=lambda x: (
            x[1],
            x[0].metadata.get("upload_date", 0)
            ),
            reverse=True
        )

        # return top_k chunks
        return [chunk for chunk, _ in ranked]        


    def query_rewriting(self, query, history):
        
        system_msg = f'''You are an excellent query_rewriter. You are provided with the user question and the history.
            Create a standalone, explicit query.
            Respond only with a short, refined question that you will use to search the Knowledge Base.
            It should be a VERY short specific question most likely to surface content. Focus on the question details.
            IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
            '''
        user_msg = f'''
            This is the history of your conversation so far with the user:
            {history}

            And this is the user's current question:
            {query}'''

        rewritten_query = self.llm2.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        return rewritten_query.choices[0].message.content

    def fetch_context_unranked(self, question, college_name, k: int = DEFAULT_RETRIEVAL_K):
        t0=time.time()
        retriever = get_college_retriever(college_name, self.client, k=10)
        t1=time.time()
        context=retriever.invoke(question)
        t2=time.time()
        chunks=[]
        print({
        "retriever_creation": t1 - t0,
        "invoke_time": t2 - t1
       })
        for chunk in context:
            chunks.append(
                Result(
                    page_content=chunk.page_content,
                    metadata=chunk.metadata,
                )
            )
        return chunks

    def fetch_context(self,college_name,original_question,history):
        start=time.time()
        rewritten_query=self.query_rewriting(original_question,history)  
        t1=time.time() 
        
        chunk2=self.fetch_context_unranked(rewritten_query,college_name)
        
        t2=time.time()
        reranked=self.rerank(rewritten_query,chunk2)
        top_semantic=reranked[:8]
        t3=time.time()
        final_context=sorted(
            top_semantic,
            key=lambda x: x.metadata.get("upload_date", "2026-01-01 00:00"),
            reverse=True
        )
        print({
            "query rewriting":t1-start,
            "fetching context unranked":t2-t1,
            "reranking":t3-t2
        })
        return final_context[:3]


    

    

      
