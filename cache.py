import redis
import hashlib
import json
import numpy as np
import os
from dotenv import load_dotenv
import chromadb
from answer_pipeline import client, CollegeChatbot, Result

chat=CollegeChatbot()

load_dotenv(override=True)

# =========================
# REDIS CLOUD CONNECTION
# =========================

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    ssl=False,              
    decode_responses=False
)

print("Redis ping:", r.ping())


# =========================
# CONFIG
# =========================

EMB_TTL = 30 * 24 * 3600
RET_TTL = 12 * 3600
RR_TTL = 12 * 3600
ANS_TTL = 3 * 24 * 3600

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 6


# =========================
# UTIL FUNCTIONS
# =========================

def sha1(text: str):
    return hashlib.sha1(text.encode()).hexdigest()


def normalize(q: str):
    return " ".join(q.lower().strip().split())


def to_bytes(vec):
    return vec.astype(np.float32).tobytes()


def from_bytes(b):
    return np.frombuffer(b, dtype=np.float32)


def get_version(key):
    v = r.get(key.encode())
    return int(v) if v else 1


# =========================
# EMBEDDING MODEL
# =========================

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    return embed_model.encode(text)




# =========================
# CACHE FUNCTIONS
# =========================

def get_embedding(question):

    q = normalize(question)
    h = sha1(q)

    emb_ver = get_version("emb:ver")

    key = f"emb:v{emb_ver}:{h}".encode()

    cached = r.get(key)

    if cached:
        return from_bytes(cached)

    vec = embed(q)

    r.setex(key, EMB_TTL, to_bytes(vec))

    return vec


def get_retrieval(question, cid):

    q = normalize(question)
    h = sha1(q)

    kb_ver = get_version(f"kb:ver{cid}")
    ret_ver = get_version("ret:ver")

    key = f"ret:kb{kb_ver}:v{ret_ver}:{h}".encode()

    cached = r.get(key)

    if cached:
        data = json.loads(cached.decode())
        return [
            Result(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]

    # Fetch fresh unranked context using the existing pipeline
    docs = chat.fetch_context_unranked(question, cid)

    payload = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]

    r.setex(key, RET_TTL, json.dumps(payload))

    return docs


def get_rerank(question, cid, history):

    # Use rewritten query for caching so history-aware semantics are cached
    rewritten = chat.query_rewriting(question, history or [])

    q = normalize(rewritten)
    h = sha1(q)

    kb_ver = get_version(f"kb:ver{cid}")
    rr_ver = get_version("rr:ver")

    key = f"rr:kb{kb_ver}:v{rr_ver}:{h}".encode()

    cached = r.get(key)

    if cached:
        data = json.loads(cached.decode())
        return [
            Result(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]

    # Cache miss: use full pipeline to fetch and rerank
    top_docs = chat.fetch_context(cid, question, history)

    payload = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in top_docs
    ]

    r.setex(key, RR_TTL, json.dumps(payload))

    return top_docs


def get_answer_cache(question, cid):

    q = normalize(question)
    h = sha1(q)

    kb_ver = get_version(f"kb:ver{cid}")
    ans_ver = get_version("ans:ver")

    key = f"ans:kb{kb_ver}:v{ans_ver}:{h}".encode()

    cached = r.get(key)

    if cached:
        return cached.decode()

    return None


def set_answer_cache(question, answer, cid):

    q = normalize(question)
    h = sha1(q)

    kb_ver = get_version(f"kb:ver{cid}")
    ans_ver = get_version("ans:ver")

    key = f"ans:kb{kb_ver}:v{ans_ver}:{h}".encode()

    r.setex(key, ANS_TTL, answer)
