import json 
import os 
import time 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))
documents=[]
with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
    data=json.load(f)
for item in data:
    text=f"""Content:{item.get("Content","")}
    """.strip()
    documents.append(Document(page_content=text,metadata={"source":item.get("source",""),
                                                     "id":item.get("id")}))
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v6/embedding+db/content/db6_v6",embedding_function=embeddings,collection_name="db6_v6")
dense_retriever=db.as_retriever(search_kwargs={"k":5})
bm25=BM25Retriever.from_documents(documents)
bm25.k=5 
hybrid=EnsembleRetriever(retrievers=[dense_retriever,bm25],
                            weights=[0.4,0.6])
Question=input("Entrez votre Question")
rewrite_prompt = f"""
You are a search query optimizer for a Moroccan government RAG system.

Rewrite the user's question into a version that is easier for retrieval.

Rules:
- Keep exactly the same meaning.
- Expand abbreviations if useful.
- Replace pronouns with explicit nouns.
- If the question is in Darija, rewrite it into Modern Standard Arabic or French while preserving meaning.
- Keep important keywords.
- Return ONLY the rewritten query.

Question:
{Question}
"""

rewritten = llm.invoke([("user", rewrite_prompt)]).content.strip()

print("Rewritten:", rewritten)
results=hybrid.invoke(rewritten)
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
pairs=[(Question,doc.page_content) for doc in results]
scores=reranker.predict(pairs)
ranked_docs=[doc for _,doc in sorted(zip (scores,results),
                                     key=lambda x:x[0],
                                     reverse=True)]
final_docs=ranked_docs[:3]
context="\n\n".join([doc.page_content for doc in final_docs])
system_prompt=f"""
You are an expert assistant for Moroccan public administration.

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
{context}
Question:
{rewritten}
"""

response=llm.invoke([("system",system_prompt),("user",user_prompt)])
print(response.content)
