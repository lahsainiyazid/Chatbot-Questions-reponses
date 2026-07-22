import os 
import json
import time 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever 
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma 
from langchain_classic.retrievers import EnsembleRetriever
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper 
from langchain_groq import ChatGroq

# 1. Setup Client
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0,api_key=os.environ.get("GROQ_API_KEY"))

# 2. Load Documents
documents = []
with open("/home/yazid/stage/v6/chunking/complete_windows.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    text = f"""
        Content: {item.get("Content", "")}
    """.strip()
    documents.append(Document(page_content=text, metadata={"source": item.get("source", ""), "id": item.get("id", "")}))

# 3. Setup Embeddings and Vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = Chroma(
    persist_directory="/home/yazid/stage/v6/embedding+db/content/db6_v6",
    embedding_function=embeddings,
    collection_name="db6_v6"
)

# 4. Initialize Retrievers and Hybrid Search (Ensemble)
vector_retriever = db.as_retriever(search_kwargs={"k":5})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5 

hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4]
)

# 5. Initialize CrossEncoder Reranker
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3") # Replace with your preferred cross-encoder model if different

# 6. RAG Function
def ask_rag(question):
    total_start=time.perf_counter()
    retrieval_start=time.perf_counter()
    # retrieval
    results = hybrid.invoke(question)
    retrieval_time=time.perf_counter()-retrieval_start
    if not results:
        return "أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال.", []

    # reranking
    reranker_start=time.perf_counter()
    pairs = [
        (question, doc.page_content)
        for doc in results
    ]

    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    final_docs = ranked_docs[:3]
    reranker_time=time.perf_counter()-reranker_start
    # contexts for RAGAS
    contexts = [
        doc.page_content
        for doc in final_docs
    ]

    context = "\n\n".join(contexts)

    prompt = f"""
Strict rules:

1. Answer only from context.
2. If answer is missing reply:
"أنا آسف، ولكن السياق المقدم لا يسمح بالإجابة على هذا السؤال."

3. Be concise and organized in Arabic.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()
    llm_start=time.perf_counter()
    response=llm.invoke(prompt)  
    llm_time=time.perf_counter()-llm_start
    total_time=time.perf_counter()-total_start
    print("\n========== Latency ==========")
    print(f"Retrieval : {retrieval_time:.3f} s")
    print(f"Reranker : {reranker_time:.3f} s")
    print(f"LLM       : {llm_time:.3f} s")
    print(f"Total     : {total_time:.3f} s")
    print("=============================\n")
    
    return response.content, contexts

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

test_results = []

MAX_SAMPLES = 3 

with open(
    "/home/yazid/stage/dataset/ragas_dataset.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for i, line in enumerate(f):

        if i >= MAX_SAMPLES:
            break

        item = json.loads(line)

        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"Testing {i+1}/{MAX_SAMPLES}")

        answer, contexts = ask_rag(question)

        test_results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

dataset = Dataset.from_list(test_results)
evaluator_llm=LangchainLLMWrapper(llm)
evaluator_embeddings=LangchainEmbeddingsWrapper(embeddings)
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

print(results)

results.to_pandas().to_csv(
    "ragas_results.csv",
    index=False
)


