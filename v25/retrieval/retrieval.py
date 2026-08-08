import json 
import os 
import time 
import hashlib
import redis 
from fastapi import FastAPI 
from pydantic import BaseModel 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
from langchain_chroma import Chroma 
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever 
from sentence_transformers import CrossEncoder 
from langchain_groq import ChatGroq 
from dotenv import load_dotenv 
load_dotenv()
app=FastAPI()
redis_url=os.getenv("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL not found!")
redis_client=redis.from_url(redis_url,
                            decode_responses=True,
                            ssl_cert_reqs="none",
                            socket_connect_timeout=2,
                            socket_timeout=2)
class QuestionRequest(BaseModel):
    question:str 
llm_expansion=ChatGroq(model="llama-3.3-8b-instant",temperature=0,api_key=os.getenv("GROQ_API_KEY"))
llm_answer=ChatGroq(model="llama-3.1-70b-versatile",temperature=0,api_key=os.getenv("GROQ_API_KEY"))
documents=[]
try:
    with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
        data=json.load(f)
        for item in data:
            text=f"""Content:{item.get("Content","")}""".strip()
            documents.append(Document(page_content=text,metadata={"source":item.get("source",""),"id":{item.get("id","")}}))
except Exception as e:
    print(f"There was an error:{e}")
embeddings=HuggingFaceEmbeddings(model_name="intflot/multilingual-e5-large",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v17/embeddings+db/content/db_17_v17",emebdding_function=embedddings,collection_name="db_17_v17")
bm25=BM25Retriever.from_documents(documents)
bm25.k=5 
dense=db.as_retriever(search_kwargs={"k":5})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],
                         weights=[0.6,0.4])
reranker=CrossEncoder("BAAI/bge-reranker-base")
@app.get("/")
def home():
    return {"Rag is running":"True",
            "Version":"V25"}
@app.post("/ask")
def ask_rag_question(request:QuestionRequest):
    start=time.time()
    normalized_question=request.question.strip().lower()
    cache_key=f"rag_cache:{hashlib.md5(normalized_question.encode().hexdigest)}"
    try:
        cached_response=redis_client.get(cache_key)
        print(f"{request.question} was found in the cache")
        return json.loads(cached_response)
    except Exception as e:
        print(f"Error:{e} while trying to load the cache")



