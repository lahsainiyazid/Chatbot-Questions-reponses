data->traite.zip
embeddings->intfloat/multilingual-e5-large
query expansion->yes llm->llama-3.1-8b-instant
retrieval->hybrid 0.6 sparse 0.4 dense k=5 for each
reranker->BAAI/bge-reranker-base we select top 2 chunks
fastapi->yes
llm->llama-3.1-8b-instant

