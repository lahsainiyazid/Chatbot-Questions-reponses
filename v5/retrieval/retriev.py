import json 
import os 
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
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v4/embeddings+db/content/db4_v4",embedding_function=embeddings,collection_name="db4_v4")
with open("/home/yazid/stage/v4/chunking/complete_data_v4.json") as f:
    data=json.load(f)
for item in data:
    text=f"""Content:{item.get("Content","")}
    """.strip()
    documents.append(Document(page_content=text,metadata={"source":item.get("source",""),
                                                          "id":item.get("id","")}))

bm25=BM25Retriever.from_documents(documents)
db.k=10 
dense=db.as_retriever(search_kwargs={"k":10})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],
                          weights=[0.7,0.3])
reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
Question=input("Entrez votre Question:")
results=hybrid.invoke(Question)
pairs=[(Question,result.page_content) for result in results]
scores=reranker.predict(pairs)
sorted_docs=[doc for _,doc in sorted(zip(scores,results),
                                     key=lambda x:x[0],
                                     reverse=True)]
final_docs=sorted_docs[:5]
context="\n\n".join([doc.page_content for doc in final_docs])
prompt =f"""
Strict rules:
1. Factuality: Rely strictly on facts directly mentioned in the context. Do not assume, extrapolate, or use external knowledge.
2. Transparency: If the answer is not in the context, reply exactly: "أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال." Do not guess.
3. Clarity: Be concise, precise, and organized in Arabic.

CONTEXT: {context}
QUESTION: {Question}""".strip()
results=client.models.generate_content(model="gemini-2.5-flash",
                                      contents=prompt)
print(results.text)
