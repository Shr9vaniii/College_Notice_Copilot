
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI
from openai import LengthFinishReasonError, BadRequestError

from multiprocessing import Pool
import os
from dotenv import load_dotenv

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = OpenAI(api_key=groq_api_key,base_url="https://api.groq.com/openai/v1")
MODEL = "openai/gpt-oss-120b"
gemini_api_key = os.getenv("GEMINI_API_KEY")
load_dotenv(override=True)
WORKERS = 4

AVERAGE_CHUNK_SIZE=500
MAX_OUTPUT_TOKENS = 3000  # Limit output to avoid hitting token limits
MAX_INPUT_CHARS = 10000  # Split documents larger than this
MIN_INPUT_CHARS = 100  # Minimum size before we stop splitting

class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def make_prompt(text):
        how_many = (len(text) // AVERAGE_CHUNK_SIZE) + 1
        # Limit how_many to prevent too many chunks
        how_many = min(how_many, 10)  # Cap at 10 chunks to avoid token limits
        
        prompt = f"""You take text and you split the text into overlapping chunks for a KnowledgeBase.
The text is a notice from a college.
You should divide up the text as you see fit, being sure that the entire text is returned across the chunks - don't leave anything out.
The text can contain text from tables, images, etc.
For tables, make sure the text is appropriately chunked to answer specific questions.
This text should probably be split into approximately {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you must provide:
- headline: A brief heading (a few words) that is most likely to be surfaced in a query
- summary: A few sentences summarizing the content to answer common questions
- original_text: The original text of this chunk from the provided document, exactly as is, not changed in any way

IMPORTANT: Make sure all chunks together represent the entire text with overlap. Do not skip any part of the text.

Here is the text:
{text}

Respond with valid JSON containing a list of chunks. Each chunk must have headline, summary, and original_text fields."""
        return prompt

def make_messages(text):
    return [
        {"role": "user", "content": make_prompt(text)},
    ]

def split_text_into_parts(text, max_chars=MAX_INPUT_CHARS):
    """Split large text into smaller parts for processing"""
    if len(text) <= max_chars:
        return [text]
    
    parts = []
    # Try to split at paragraph boundaries
    paragraphs = text.split('\n\n')
    current_part = ""
    
    for para in paragraphs:
        if len(current_part) + len(para) + 2 <= max_chars:
            current_part += para + "\n\n" if current_part else para
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = para
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

def process_document(document, recursion_depth=0):
    """Process a document, handling large documents by splitting them"""
    MAX_RECURSION_DEPTH = 3  # Prevent infinite recursion
    
    # Safety check: prevent infinite recursion
    if recursion_depth >= MAX_RECURSION_DEPTH:
        print(f"Max recursion depth reached. Processing document as-is (length: {len(document)})")
        # Try with a very simple prompt for small documents
        if len(document) < MIN_INPUT_CHARS:
            # For very small documents, create a single chunk manually
            return Chunks(chunks=[Chunk(
                headline="Document",
                summary=document[:200] + "..." if len(document) > 200 else document,
                original_text=document
            )])
    
    # If document is too large, split it
    if len(document) > MAX_INPUT_CHARS:
        parts = split_text_into_parts(document)
        all_chunks = []
        
        for part in parts:
            try:
                chunks = process_document(part, recursion_depth + 1)
                if chunks and hasattr(chunks, 'chunks'):
                    all_chunks.extend(chunks.chunks)
                elif isinstance(chunks, Chunks):
                    all_chunks.extend(chunks.chunks)
            except Exception as e:
                print(f"Error processing part: {e}")
                continue
        
        # Combine all chunks into a single Chunks object
        return Chunks(chunks=all_chunks)
    
    # Process normal-sized document
    messages = make_messages(document)
    try:
        response = groq_client.chat.completions.parse(
            messages=messages,
            response_format=Chunks,
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        return response.choices[0].message.content
    except (LengthFinishReasonError, BadRequestError) as e:
        # If hitting token limit or JSON validation error, split the document further
        error_type = "token limit" if isinstance(e, LengthFinishReasonError) else "JSON validation"
        print(f"Hit {error_type} error, splitting document further (current length: {len(document)})...")
        
        # Don't split if document is already very small
        if len(document) < MIN_INPUT_CHARS:
            print(f"Document too small to split ({len(document)} chars). Creating single chunk manually.")
            return Chunks(chunks=[Chunk(
                headline="Document",
                summary=document[:200] + "..." if len(document) > 200 else document,
                original_text=document
            )])
        
        parts = split_text_into_parts(document, max_chars=MAX_INPUT_CHARS // 2)
        all_chunks = []
        
        for part in parts:
            try:
                chunks = process_document(part, recursion_depth + 1)
                if chunks and hasattr(chunks, 'chunks'):
                    all_chunks.extend(chunks.chunks)
                elif isinstance(chunks, Chunks):
                    all_chunks.extend(chunks.chunks)
            except Exception as err:
                print(f"Error processing part after retry: {err}")
                continue
        
        if all_chunks:
            return Chunks(chunks=all_chunks)
        else:
            # Fallback: try with even smaller chunks
            if len(document) >= MIN_INPUT_CHARS * 2:
                print("Trying with even smaller chunks...")
                parts = split_text_into_parts(document, max_chars=MAX_INPUT_CHARS // 4)
                all_chunks = []
                for part in parts:
                    try:
                        chunks = process_document(part, recursion_depth + 1)
                        if chunks and hasattr(chunks, 'chunks'):
                            all_chunks.extend(chunks.chunks)
                        elif isinstance(chunks, Chunks):
                            all_chunks.extend(chunks.chunks)
                    except Exception as err:
                        print(f"Error processing small part: {err}")
                        continue
                return Chunks(chunks=all_chunks) if all_chunks else Chunks(chunks=[])
            else:
                # Last resort: create a single chunk
                return Chunks(chunks=[Chunk(
                    headline="Document",
                    summary=document[:200] + "..." if len(document) > 200 else document,
                    original_text=document
                )])
    except Exception as e:
        print(f"Error processing document: {e}")
        # Try splitting as fallback if document is large enough
        if len(document) > MIN_INPUT_CHARS:
            print("Attempting to split document as fallback...")
            parts = split_text_into_parts(document, max_chars=MAX_INPUT_CHARS // 2)
            all_chunks = []
            for part in parts:
                try:
                    chunks = process_document(part, recursion_depth + 1)
                    if chunks and hasattr(chunks, 'chunks'):
                        all_chunks.extend(chunks.chunks)
                    elif isinstance(chunks, Chunks):
                        all_chunks.extend(chunks.chunks)
                except Exception as err:
                    print(f"Error in fallback processing: {err}")
                    continue
            return Chunks(chunks=all_chunks) if all_chunks else Chunks(chunks=[])
        else:
            # For very small documents, create a single chunk
            return Chunks(chunks=[Chunk(
                headline="Document",
                summary=document[:200] + "..." if len(document) > 200 else document,
                original_text=document
            )])

def create_chunks(text):
    return process_document(text)