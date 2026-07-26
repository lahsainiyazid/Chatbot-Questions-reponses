import os 
import json 
import time 
from fastapi import FastAPI 
from pydantic import BaseModel 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever 
from sentence_transformers import CrossEncoder 
from langchain_groq import ChatGroq 

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

documents = []
try:
    with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
        data = json.load(f)
    for item in data:
        # Ensure the key matches your JSON structure (e.g., "content" vs "Content")
        content_text = item.get("Content", "") or item.get("content", "")
        text = f"Content: {content_text}".strip()
        
        documents.append(Document(
            page_content=text,
            metadata={
                "source": item.get("source", ""),
                "id": item.get("id", "")
            }
        ))
except FileNotFoundError:
    print("Warning: JSON file not found. Starting with empty documents.")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory="/home/yazid/stage/v17/embeddings+db/content/db_17_v17",
    embedding_function=embeddings,
    collection_name="db_17_v17"
)

bm25 = BM25Retriever.from_documents(documents)
bm25.k = 5 
dense = db.as_retriever(search_kwargs={"k": 5})
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.6, 0.4])
reranker = CrossEncoder("BAAI/bge-reranker-base")

@app.get("/")
def home():
    return {"Rag api is running": "True", "model": "v19"}

@app.post("/ask")
def ask_rag_question(request: QuestionRequest):
    start_time = time.time()
    
    # 1. Query Expansion
    expansion_start = time.time()
    expansion_prompt = f"""
Expand the following search query with related keywords and synonyms useful for document retrieval.
Keep the original query.
Return ONLY the expanded query.

Query:
{request.question}
"""
    expanded_query = llm.invoke(expansion_prompt).content.strip()
    expansion_time = time.time() - expansion_start
    
    # 2. Retrieval
    retrieval_start = time.time()
    results = hybrid.invoke(expanded_query)
    retrieval_time = time.time() - retrieval_start
    
    # 3. Reranking
    reranker_start = time.time()
    pairs = [(expanded_query, doc.page_content) for doc in results]
    scores = reranker.predict(pairs)
    
    # Sort by score descending
    sorted_docs = [doc for _, doc in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
    final_docs = sorted_docs[:3]
    reranker_time = time.time() - reranker_start  # Fixed: Calculate reranker_time
    
    contexts_texts = "\n\n".join([doc.page_content for doc in final_docs])
    
    # 4. Generation
    system_prompt = """You are an expert assistant for Moroccan public administration.
Answer ONLY from the provided context.
Rules:
- Never use external knowledge.
- If the answer cannot be fully supported by the context, clearly say so.
- Reply in the user's language.
"""
    user_prompt = f"""
Context:
{contexts_texts}

Question:
{request.question}
"""
    llm_start = time.time()
    response = llm.invoke([("system", system_prompt), ("user", user_prompt)])
    llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time
    
    # Safe token usage extraction
    token_usage = {}
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        um= response.response_metadata['token_usage']
        token_usage={
            "total_tokens":um.get("total_tokens",0),
            "prompt_tokens":um.get("prompt_tokens",0),
            "completion_tokens":um.get('completion_tokens',0)
        }
    elif hasattr(response, 'usage_metadata'):
        um = response.usage_metadata
        token_usage = {
            "total_tokens": um.get("total_tokens", 0),
            "prompt_tokens": um.get("input_tokens", 0),
            "completion_tokens": um.get("output_tokens", 0)
        }

    return {
        "Question": request.question,
        "Answer": response.content.strip(),
        "total_time": round(total_time, 3),
        "expansion_time": round(expansion_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "reranker_time": round(reranker_time, 3),
        "llm_time": round(llm_time, 3),
        "token_usage": {
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0)
        }
    }

