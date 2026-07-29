import os 
import json 
import time 
from fastapi import FastAPI 
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
from langchain_chroma import Chroma 
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder 
from langchain_ollama import ChatOllama 
app=FastAPI()
class QuestionRequest(BaseModel):
    question:str 
llm_expansion=ChatOllama(model="qwen2.5:3b",temperature=0,num_predict=80)
llm_answer=ChatOllama(model="qwen2.5:3b",temperature=0,num_predict=300)
documents=[]
try:
    with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
        data=json.load(f)
    for item in data:
        text=f"""
           Content:{item.get("Content","")}
        """.strip()
        documents.append(Document(page_content=text,metadata={"source":item.get("source",""),"id":item.get("id")}))
except Exception as e:
    print(f"There was error while loading chunks ,error:{e}")
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
    return{"Rag running":"True",
           "Version":"v23"}
@app.post("/ask")
def ask_rag_question(request:QuestionRequest):
    start=time.time()
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
    return{"Question":request.question,
           "Answer":answer.content.strip(),
           "total_time":round(total_time,3),
           "expansion_time":round(expansion_time,3),
           "retrieval_time":round(retrieval_time,3),
           "reranker_time":round(reranker_time,3),
            "llm_time":round(llm_time,3)}


