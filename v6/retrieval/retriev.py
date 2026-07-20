import os 
import json 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever 
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever
from google import genai 
key=os.environ.get("key")
client=genai.Client(api_key=key)
documents=[]
with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
    data=json.load(f)
for item in data:
    text=f"""
        Content:{item.get("Content","")}
    """.strip()
    documents.append(Document(page_content=text,metadata={"source":item.get("source",""),"id":item.get("id","")}))
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v6/embedding+db/content/db6_v6",embedding_function=embeddings,collection_name="db6_v6")
Question=input("Entrez votre Question:")
bm25=BM25Retriever.from_documents(documents)
bm25.k=5 
dense=db.as_retriever(search_kwargs={"k":5})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],
                         weights=[0.7,0.3])
results=hybrid.invoke(Question)
reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs=[(Question,doc.page_content) for doc in results]
scores=reranker.predict(pairs)
ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                         key=lambda x:x[0],
                                         reverse=True)]
final_docs=ranked_docs[:3]
context="\n\n".join(doc.page_content for doc in final_docs)
prompt = f"""
Strict rules:
1. Factuality: Rely strictly on facts directly mentioned in the context. Do not assume, extrapolate, or use external knowledge.
2. Transparency: If the answer is not in the context, reply exactly: "أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال." Do not guess.
3. Clarity: Be concise, precise, and organized in Arabic.

CONTEXT: {context}
QUESTION: {Question}""".strip()
response=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
print(response.text)


