
import os 
import json
import time 
from fastapi import FastAPI  
from pydantic import BaseModel  
from langchain_core.documents  import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever 
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq 
app=FastAPI()
class QuestionRequest(BaseModel):
    question:str 
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))
documents=[]
with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
    data=json.load(f)
for item in data:
    text=f"""
    Content:{item.get("Content","")}
    """.strip()
    documents.append(Document(page_content=text,metadata={"source":item.get("source"),"id":item.get("id")}))
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v6/embedding+db/content/db6_v6",embedding_function=embeddings,collection_name="db6_v6")
bm25=BM25Retriever.from_documents(documents)
bm25.k=4 
dense=db.as_retriever(search_kwargs={"k":4})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],weights=[0.6,0.4])
reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
@app.get("/")
def home():
    return{"status":"running",
           "message":"rag api is online "}
@app.post("/ask")
def ask_rag_question(request:QuestionRequest):
    start_time=time.time()
    retrieval_start=time.time()
    results=hybrid.invoke(request.question)
    retrieval_time=time.time()-retrieval_start
    reranker_start=time.time()
    pairs=[(request.question,doc.page_content) for doc in results]
    scores=reranker.predict(pairs)
    ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                         key=lambda x:x[0],
                                         reverse=True)]
    reranker_time=time.time()-reranker_start
    final_docs=ranked_docs[:3]
    contexts=[doc.page_content for doc in final_docs]
    contexts_text="\n\n".join(contexts)
    system_prompt=f""" You are an expert assistant for Moroccan public administration.
Answer ONLY from the provided context.
Before writing the final answer, internally perform these steps:
- Identify the passages relevant to the question.
- Ignore unrelated information.
- Verify that every statement in your answer is supported by the context.
- If any statement cannot be supported, remove it.
- Merge duplicate information.
- Preserve official names exactly.
Rules:
- Never use external knowledge.
- Never guess.
- Never fabricate missing information.
- If the answer cannot be fully supported by the context, clearly say so.
- Reply in the user's language.
- Be clear and professional.
"""
    user_prompt=f"""
    Context:
    {contexts_text}
    Question:
    {request.question}
"""
    llm_start_time=time.time()
    response=llm.invoke([("system",system_prompt),("user",user_prompt)])
    llm_time=time.time()-llm_start_time 
    total_time=time.time()-start_time 
    token_usage=response.response_metadata["token_usage"]
    return {"question":request.question,
            "answer":response.content.strip(),
            "total_time":round(total_time,3),
            "retrieval_time":round(retrieval_time,3),
            "reranker-time":round(reranker_time,3),
            "llm_time":round(llm_time,3),
            "token_usage":{
            "total_tokens":token_usage["total_tokens"],
            "prompt_tokens":token_usage["prompt_tokens"],
            "completion_tokens":token_usage["completion_tokens"]
            }}


