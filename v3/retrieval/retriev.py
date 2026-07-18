import os 
import json
import time 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever 
from google import genai 
key=os.environ.get("key")
client=genai.Client(api_key=key)
documents=[]
total_start=time.perf_counter()
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v3/embeddings+db/content/db3_v3",embedding_function=embeddings,collection_name="db3_v3")
with open("/home/yazid/stage/v3/chunking/complete_data_v3.json") as f:
    data=json.load(f)
for item in data:
    text=f"""
    Content={item.get("Content","")}
    """.strip()
    documents.append(Document(page_content=text,metadata={"source":item.get("source",""),
        "id":item.get("id","")}))
Question=input("Entrez votre Question:")
retrieval_start=time.perf_counter()
bm25=BM25Retriever.from_documents(documents)
bm25.k=5 
dense=db.as_retriever(search_kwargs={"k":5})
hybrid=EnsembleRetriever(retrievers=[dense,bm25],weights=[0.3,0.7])
results=hybrid.invoke(Question)
retrieval_finish=time.perf_counter()-retrieval_start
reranker_start=time.perf_counter()
reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs=[(Question,result.page_content) for result in results]
scores=reranker.predict(pairs)
ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                     key=lambda x:x[0],
                                     reverse=True)]
final_docs=ranked_docs[:3]
reranker_finish=time.perf_counter()-retrieval_start
context="\n\n".join([doc.page_content for doc in final_docs]).strip()
prompt =f"""
Strict rules:
1. Factuality: Rely strictly on facts directly mentioned in the context. Do not assume, extrapolate, or use external knowledge.
2. Transparency: If the answer is not in the context, reply exactly: "أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال." Do not guess.
3. Clarity: Be concise, precise, and organized in Arabic.

CONTEXT: {context}
QUESTION: {Question}""".strip()
llm_start=time.perf_counter()
response=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
llm_finish=time.perf_counter()-llm_start
print(f"Retrieval time :{retrieval_finish}")
print(f"Reranker time:{reranker_finish}")
print(f"LLM time:{llm_finish}")
print(response.text)
