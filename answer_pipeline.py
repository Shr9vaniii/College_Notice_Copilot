import os
from dotenv import load_dotenv
from chromadb import CloudClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ingestion import get_college_retriever
from pydantic import BaseModel,Field
from openai import OpenAI

load_dotenv(override=True)

chroma_api=os.getenv('CHROMA_API_KEY')

client = CloudClient(
    api_key=chroma_api,
    tenant='246553cc-048a-47fa-aa2a-cf9d61280656',
    database='Project'
    )

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash"

llm=ChatGoogleGenerativeAI(model=MODEL,temperature=0)

llm2=OpenAI(api_key=gemini_api_key,base_url=gemini_url)
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

    def rerank(self,question, chunks):
        system_prompt = """
    You are a document re-ranker.
    You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
    The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
    You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
    Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
    """
        user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, chunk in enumerate(chunks):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = llm2.chat.completions.parse(model=MODEL,messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
        return [chunks[i - 1] for i in order]        


    def query_rewriting(self, query, history=[]):
            system_msg=f'''You are an excellent query_rewriter. You are provided with the user question and the history.
            Create a standalone, explicit query.
            Respond only with a short, refined question that you will use to search the Knowledge Base.
            It should be a VERY short specific question most likely to surface content. Focus on the question details.
            IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
            '''
            user_msg=f'''
            This is the history of your conversation so far with the user:
            {history}

            And this is the user's current question:
            {query}'''

            rewritten_query=self.llm.invoke([SystemMessage(content=system_msg),HumanMessage(content=user_msg)])
            return rewritten_query.content  

    def fetch_context_unranked(self, question, college_name, k: int = DEFAULT_RETRIEVAL_K):
        retriever = get_college_retriever(college_name, self.client, k=k)
        context=retriever.invoke(question)
        chunks=[]
        for chunk in context:
            chunks.append(
                Result(
                    page_content=chunk.page_content,
                    metadata=chunk.metadata,
                )
            )
        return chunks

    def fetch_context(self,college_name,original_question,history):
        rewritten_query=self.query_rewriting(original_question,history)   
        chunk1=self.fetch_context_unranked(original_question,college_name)
        chunk2=self.fetch_context_unranked(rewritten_query,college_name)
        merged=self.merge_chunks(chunk1,chunk2)
        reranked=self.rerank(rewritten_query,merged)
        return reranked[:FINAL_K]


    

    

      