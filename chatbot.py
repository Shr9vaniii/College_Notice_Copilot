from cache import chat, get_answer_cache, get_rerank, set_answer_cache
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
from groq import Groq
import asyncio
import os


SYSTEM_PREVIEW_MSG=system_prompt = """
You are the CampusSync Assistant that answers the query to students of the college. Use the provided context to answer questions.
The context is sorted by RECENCY (the most recent notices appear first).
If you find conflicting information (like different dates for the same event), 
prioritize the information from the most recent notice.If you don't know, say so.
Respond only with short and relevant response.Do not respond with your thinking, respond only with the final response.If the usser greets you, greet them back.
Here is the context:
{context}
"""

semaphore=asyncio.Semaphore(3)
load_dotenv(override=True)
groq_client=Groq()
MODEL="gemini-2.5-pro"
llm=ChatGoogleGenerativeAI(model=MODEL,temperature=0)


async def question_answer(query, history, college_name):
        start=time.time()
        rewritten_query=await asyncio.to_thread(chat.query_rewriting,query,history)
        
        cached = get_answer_cache(rewritten_query, college_name)
        if cached:
            history = history.copy()
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": cached})
            
            return history
        
        # get_rerank will internally rewrite the query and cache by rewritten form
        
        context = await asyncio.to_thread(get_rerank,rewritten_query, college_name, history)
        t1=time.time()
        text = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in context])
        system_prompt = SYSTEM_PREVIEW_MSG.format(context=text)

        converted=convert_to_groq_messages(system_prompt,history,rewritten_query)
        async with semaphore:

            response =await asyncio.to_thread(groq_client.chat.completions.create,model="openai/gpt-oss-20b",  # fast model
             messages=converted)
        t2=time.time()
        history.append({'role': 'user', "content": query})
        history.append({'role': 'assistant', "content": response.choices[0].message.content})
        set_answer_cache(rewritten_query, response.choices[0].message.content, college_name)

        print({
            "retrieval_and_reranking":t1-start,
            "llm":t2-t1
        })
        
        return history

async def api_question_answer(query: str, college_name: str = "default"):
    history = [] 

    result = await question_answer(query, history, college_name)

    # result is full history → we return only last answer
    if result and isinstance(result, list):
        return result[-1]["content"]

    return result     

def convert_to_groq_messages(system_prompt, history, query):
    messages = []

    # system message
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # history
    for h in history:
        if h["role"] == "user":
            messages.append({"role": "user", "content": h["content"]})
        elif h["role"] == "assistant":
            messages.append({"role": "assistant", "content": h["content"]})

    # current query
    messages.append({
        "role": "user",
        "content": query
    })

    return messages       
