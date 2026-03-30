from cache import chat, get_answer_cache, get_rerank, set_answer_cache
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time


SYSTEM_PREVIEW_MSG="""You are a helpful,friendly assistant for students to get their college notices.
    You are chatting with students.
    Use relevant context to generate response
    If you don't know, say so.
    Relevant context:
    {context}
    
    """

load_dotenv(override=True)
MODEL="gemini-2.5-flash"
llm=ChatGoogleGenerativeAI(model=MODEL,temperature=0)


def question_answer(query, history, college_name):
        start=time.time()
        
        cached = get_answer_cache(query, college_name)
        if cached:
            print("Answer cache hit")
            return cached
        
        # get_rerank will internally rewrite the query and cache by rewritten form
        rewritten_query=chat.query_rewriting(query)
        context = get_rerank(query, college_name, history)
        t1=time.time()
        text = "\n\n".join(doc.page_content for doc in context)
        system_prompt = SYSTEM_PREVIEW_MSG.format(context=text)

        msg=[SystemMessage(content=system_prompt),
        *chat.gradio_history_to_messages(history),
        HumanMessage(content=query)]
        
        
        response =llm.invoke(msg)
        t2=time.time()
        history.append({'role': 'user', "content": query})
        history.append({'role': 'assistant', "content": response.content})
        set_answer_cache(rewritten_query, response.content, college_name)

        print({
            "retrieval_and_reranking":t1-start,
            "llm":t2-t1
        })
        
        return history

def api_question_answer(query: str, college_name: str = "default"):
    history = [] 

    result = question_answer(query, history, college_name)

    # result is full history → we return only last answer
    if result and isinstance(result, list):
        return result[-1]["content"]

    return result        