import os
import json
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(
    os.getenv("MONGODB_URI"),
    username=os.getenv("MONGODB_USERNAME"),
    password=os.getenv("MONGODB_PASSWORD")
)
collection = client["rag_db"]["chunks"]

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

with open("/home/yazid/stage/v6/chunking/complete_windows.json") as f:
    data = json.load(f)

documents = [
    Document(
        page_content=item.get("Content", ""),
        metadata={"source": item.get("source", ""), "id": item.get("id", "")}
    )
    for item in data
]

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index"
)
vector_store.add_documents(documents)
print(f"Inserted {len(documents)} documents.")

existing = list(collection.list_search_indexes())
if any(idx["name"] == "vector_index" for idx in existing):
    print("Index already exists.")
else:
    index_model = SearchIndexModel(
        definition={
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"}
            ]
        },
        name="vector_index",
        type="vectorSearch"
    )
    collection.create_search_index(model=index_model)
    print("Index creation started — takes ~30-60s to become queryable.")
