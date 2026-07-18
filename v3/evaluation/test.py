from pathlib import Path 
import pandas as pd 
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator 
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper 
from ragas import evaluate 
from ragas.metrics import (faithfulness,answer_relevancy,context_precision,context_recall)
from langchain_google_genai import ChatGoogleGenerativeAI 

documents=[]
folder1=Path("/home/yazid/stage/data/traite/documents")
folder2=Path("/home/yazid/stage/data/traite2")
folders=[folder1,folder2]
for folder in folders:
    for file in folder.glob("*.docx"):
        loader=Docx2txtLoader(str(file))
        documents.extend(loader.load())


print(f"Number of docs:{len(documents)}")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
generator=TestsetGenerator(llm=LangchainLLMWrapper(llm),
                           embedding_model=LangchainEmbeddingsWrapper(embeddings))
testset=generator.generate_with_langchain_docs(documents,testset_size=50)
df=testset.to_pandas()
df.to_csv("testset.csv",index=False)
