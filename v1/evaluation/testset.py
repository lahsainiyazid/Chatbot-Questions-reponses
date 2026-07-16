import os 
import json  
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper 
from ragas.testset import TestsetGenerator 
from ragas.testset.synthesizers import default_query_distribution
key=os.environ.get("key")
with open("/home/yazid/stage/v1/chunking/chunks_questions_v1.json","r",encoding="utf-8") as f:
    data=json.load(f)
docs=[Document(page_content=item["page_content"],metadata=item.get("metadata",{})) for item in data]
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                           google_api_key=key,
                           temperature=0)
generator_llm=LangchainLLMWrapper(llm)
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
generator_embeddings=LangchainEmbeddingsWrapper(embeddings)
generator=TestsetGenerator(llm=generator_llm,
                           embedding_model=generator_embeddings)
query_distribution=default_query_distribution(generator_llm)
testset=generator.generate_with_langchain_docs(docs,testset_size=20,query_distribution=query_distribution)
df=testset.to_pandas()
df.to_json("ragas_test_set_v1.json",orient="records",forse_ascii=False,indent=4)
print("Successfully created the test set")

