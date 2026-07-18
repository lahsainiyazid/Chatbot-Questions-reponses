
import json 
import os 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever 
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever
from google import genai
import time 

# Start total execution time
total_start = time.perf_counter()

# Initialize API client
key = os.environ.get("key")
client = genai.Client(api_key=key)

# Load data and create documents first (before embeddings)
documents = []
with open("/home/yazid/stage/v4/chunking/complete_data_v4.json") as f:
    data = json.load(f)

for item in data:
    text = f"""Content:{item.get("Content","")}
    """.strip()
    documents.append(Document(
        page_content=text,
        metadata={
            "source": item.get("source",""),
            "id": item.get("id","")
        }
    ))

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

db = Chroma(
    persist_directory="/home/yazid/stage/v4/embeddings+db/content/db4_v4",
    embedding_function=embeddings,
    collection_name="db4_v4"
)

# Get user input before starting retrieval
Question = input("Entrez votre Question:")

# Start retrieval time measurement
retrieval_start = time.perf_counter()

# Setup retrievers
bm25 = BM25Retriever.from_documents(documents)
db.k = 5
dense = db.as_retriever(search_kwargs={"k": 5})
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.7, 0.3]
)

# Perform retrieval
results = hybrid.invoke(Question)
retrieval_finish = time.perf_counter() - retrieval_start

# Reranking
reranker_start=time.perf_counter()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(Question, result.page_content) for result in results]
scores = reranker.predict(pairs)
sorted_docs = [doc for _, doc in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
final_docs = sorted_docs[:5]
reranker_finish=time.perf_counter()-reranker_start
# Prepare context and prompt
context = "\n\n".join([doc.page_content for doc in final_docs])
prompt = f"""
Strict rules:
1. Factuality: Rely strictly on facts directly mentioned in the context. Do not assume, extrapolate, or use external knowledge.
2. Transparency: If the answer is not in the context, reply exactly: "أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال." Do not guess.
3. Clarity: Be concise, precise, and organized in Arabic.

CONTEXT: {context}
QUESTION: {Question}""".strip()

# Start LLM generation time measurement
llm_start = time.perf_counter()
results = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
llm_finish = time.perf_counter() - llm_start

# Print results and metrics
print(results.text)
print(f"Retrieval time: {retrieval_finish:.4f} seconds")
print(f"LLM Time: {llm_finish:.4f} seconds")
print(f"Reranker time:{retrieval_finish:.4f}seconds ")
# Calculate total time
total_finish = time.perf_counter() - total_start
print(f"Total execution time: {total_finish:.4f} seconds")

