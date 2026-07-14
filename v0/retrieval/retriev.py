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
question=input("Entrez votre question:")
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
db=Chroma(persist_directory="/home/yazid/stage/v0/embeddings+db/db_questions_0",embedding_function=embeddings,collection_name="db_questions_0")
documents=[]
with open("/home/yazid/stage/v0/chunking/chunks_questions.json","r",encoding="utf-8") as f:
    data=json.load(f)
for item in data:
    text=f"""
 Content:{item.get("page_content","")}
    """.strip()
    documents.append(Document(page_content=text,
                              metadata={"file_name":item.get("file_name",""),
        "source":item.get("source",""), 
"page":item.get("page",""),
"folder_date":item.get("folder_date",""),
"start_index":item.get("start_index","")}))
bm25=BM25Retriever.from_documents(documents)
bm25.k=10
dense=db.as_retriever(search_kwargs={"k":10})
hybrid=EnsembleRetriever(retrievers=[bm25,dense],
                          weights=[0.7,0.3])
results=hybrid.invoke(question)
reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs=[(question,doc.page_content) for doc in results]
scores=reranker.predict(pairs)
ranked_docs=[doc for _,doc in sorted(zip(scores,results),
                                     key=lambda x:x[0],
                                     reverse=True)]
final_docs=ranked_docs[:3]
context="\n\n".join([doc.page_content for doc in final_docs])
prompt=f"""Vous êtes un assistant virtuel expert. Votre tâche est de répondre à la QUESTION de l'utilisateur en vous basant TOUJOURS et UNIQUEMENT sur le CONTEXTE fourni ci-dessous.

Voici les règles strictes à respecter :
1. Factualité : Appuyez-vous uniquement sur les faits directement mentionnés dans le contexte. N'inventez rien, n'extrapolez pas et n'utilisez pas vos connaissances externes.
2. Transparence : Si la réponse à la question ne se trouve pas dans le contexte, répondez exactement : "Je suis désolé, mais le contexte fourni ne permet pas de répondre à cette question." Ne tentez pas de deviner.
3. Clarté : Soyez concis, précis et structurez votre réponse si nécessaire.
CONTEXTE :{context}
QUESTION :{question}
    """
response=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
print(response.text)


