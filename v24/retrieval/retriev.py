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
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq 
app=FastAPI()
redis_client=redis.Redis(host=os.environ.get("REDIS_HOST","localhost"),
                         port=int(os.environ.get("REDIS_PORT",6379)),
                         db=int(os.environ.get("REDIS_DB",0)),
                          decode_responses=True,
                        socket_connect_timeout=2,
                        socket_timeout=2)
class QuestionRequest(BaseModel):
    question:str
llm_expansion=ChatGroq(model="llama-3.1-8b-instant",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))
llm_answer=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))
documents=[]
try:
    with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
        data=json.load(f)
    for item in data:
        text=f"""
         Content:{item.get("Content","")}
        """.strip()
        documents.append(Document(page_content=text,metadata={"source":item.get("source",""),
                                                          "id":item.get("id","")}))
except Exception as e:
    print(f"There was an error error-code:{e}")
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v17/embeddings+db/content/db_17_v17",embedding_function=embeddings,collection_name="db_17_v17")
bm25=BM25Retriever.from_documents(documents)
bm25.k=5 
dense=db.as_retriever(search_kwargs={"k":5})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],
                         weights=[0.6,0.4])
reranker=CrossEncoder("BAAI/bge-reranker-base")
@app.get("/")
def home():
        return{"Rag is running":"True",
               "Version":"V24"}
@app.post("/ask")
def ask_rag_question(request:QuestionRequest):
    start=time.time()
    normalized_question=request.question.strip().lower();
    cache_key=f"rag_cache:{hashlib.md5(normalized_question.encode('utf-8')).hexdigest()}"
    try:
        cached_response=redis_client.get(cache_key)
        if cached_response:
            print(f"{request.question} was found in the cache:")
            return json.loads(cached_response)
    except Exception as e:
        print(f"Error:{e}")
    expansion_start=time.time()
    expansion_prompt = f"""أنت محرك بحث ذكي متخصص في الوثائق الإدارية. مهمتك هي إعادة صياغة السؤال التالي إلى استعلامين بديلين دقيقين لتحسين البحث.

قواعد صارمة وممنوع مخالفتها:
1. أخرج الاستعلامين فقط، ولا شيء غيرهما.
2. ممنوع منعاً باتاً إضافة أي مقدمات أو خواتم أو شروحات.
3. ممنوع استخدام الأرقام أو النقاط (مثل: 1- أو *).
4. افصل بين الاستعلامين بسطر جديد واحد فقط.
5. استخدم مصطلحات إدارية رسمية ودقيقة.

السؤال: {request.question}

الاستعلامان:"""
    expanded_query=llm_expansion.invoke(expansion_prompt).content.strip()
    expansion_time=time.time()-expansion_start
    retrieval_start=time.time()
    results=hybrid.invoke(expanded_query)
    retrieval_time=time.time()-retrieval_start
    reranker_start=time.time()
    pairs=[(request.question,doc.page_content) for doc in results]
    scores=reranker.predict(pairs)
    ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                         key=lambda x:x[0],
                                         reverse=True)]
    final_docs=ranked_docs[:2]
    reranker_time=time.time()-reranker_start
    context="\n\n".join(doc.page_content for doc in final_docs)
    system_prompt = """You are an expert assistant for Moroccan public administration.
Answer ONLY from the provided context.
Rules:
- Never use external knowledge.
- If the answer cannot be fully supported by the context, clearly say so.
- Reply in the user's language.
"""
    user_prompt = f"""
Context:
{context}

Question:
{request.question}
"""
    llm_start=time.time()
    answer=llm_answer.invoke([("system",system_prompt),("human",user_prompt)])
    llm_time=time.time()-llm_start 
    total_time=time.time()-start 
    token_usage=answer.response_metadata["token_usage"]
    final_response={"Question":request.question,
           "Answer":answer.content.strip(),
           "total_time":round(total_time,3),
           "expansion_time":round(expansion_time,3),
           "retrieval_time":round(retrieval_time,3),
           "reranker_time":round(reranker_time,3),
            "llm_time":round(llm_time,3),
"token_usage":{"total_tokens":token_usage["total_tokens"],
               "prompt_tokens":token_usage["prompt_tokens"],
               "completion_tokens":token_usage["completion_tokens"]},
        "cached":"False"}
    try:
        redis_client.setex(cache_key,86400,json.dumps(final_response))
    except Exception as e:
        print(f"Failed to save to cache error:{e}")
    return final_response


    


        

    

