import os 
import json 
import time 
from fastapi  import FastAPI 
from pydantic import BaseModel 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
from langchain_chroma import Chroma 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from sentence_transformers import CrossEncoder 
from langchain_groq import ChatGroq 
app=FastAPI()
class QuestionRequest(BaseModel):
    question:str 
llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))
documents=[]
try:
    with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
        data=json.load(f)
    documents=[]
    for item in data:
        text=f"""
         Content:{item.get("Content","")}
          """.strip()
        documents.append(Document(page_content=text,metadata={"source":item.get("source",""),"id":item.get("id","")}))
except Exception as e:
    print(f"Error:{e}")
embeddings=HuggingFaceEmbeddings(model="intfloat/multilingual-e5-large",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
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
           "model":"v20"}
@app.post("/ask")
def ask_rag_question(request:QuestionRequest):
    start=time.time()
    expansion_start=time.time()
    EXPANSION_PROMPT = f"""أنت خبير في معالجة اللغة العربية واسترجاع المعلومات. مهمتك هي توسيع استعلام المستخدم لتحسين نتائج البحث في نظام RAG.

القواعد:
1. قم بتوليد 8-10 مصطلحات أو عبارات بحثية إضافية ذات صلة وثيقة بالاستعلام الأصلي
2. أضف مرادفات مختلفة (فصحى وعامية إن أمكن)
3. أضف مفاهيم أوسع (تعميم) ومفاهيم أضيق (تخصيص)
4. أضف صيغاً مختلفة للكلمات (مشتقات، جمع، مفرد)
5. ركز على المصطلحات التي من المرجح أن تظهر في الوثائق المستهدفة
6. تأكد من أن جميع المصطلحات المضافة تحافظ على النية الأصلية للاستعلام

الاستعلام الأصلي: {request.question}

مصطلحات البحث الموسعة (اكتب فقط المصطلحات في سطر واحد، مفصولة بفواصل، بدون أرقام أو تفسيرات):
"""
    expanded_prompt=llm.invoke(EXPANSION_PROMPT).content.strip()
    expansion_time=time.time()-expansion_start
    retrieval_start=time.time()
    results=hybrid.invoke(expanded_prompt)
    retrieval_time=time.time()-retrieval_start
    reranker_start=time.time()
    pairs=[(request.question,doc.page_content) for doc in results]
    scores=reranker.predict(pairs)
    ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                          key=lambda x:x[0],
                                          reverse=True)]
    reranker_time=time.time()-reranker_start
    final_docs=ranked_docs[:3]
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
    response=llm.invoke([("system",system_prompt),("user",user_prompt)])
    llm_time=time.time()-llm_start
    total_time=time.time()-start
    token_usage=response.response_metadata["token_usage"]
    return {
        "Question":request.question,
    "Answer":response.content.strip(),
        "total_time":round(total_time,3),
        "retrieval_time":round(retrieval_time,3),
        "reranker_time":round(reranker_time,3),
        "expansion_time":round(expansion_time,3),
        "llm_time":round(llm_time,3),
        "token_usage":{
            "total_tokens":token_usage["total_tokens"],
            "prompt_tokens":token_usage["prompt_tokens"],
            "completion_tokens":token_usage["completion_tokens"]
        }
    }



