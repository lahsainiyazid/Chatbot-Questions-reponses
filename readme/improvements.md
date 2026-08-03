Based on the configuration in `info.md` and the performance metrics in `tests.md`, your RAG pipeline is already well-structured (using hybrid search, reranking, and query expansion). However, there is a clear bottleneck: **the reranker is taking the longest time (1.7s to 2.8s)**, and some components are not fully optimized for the Arabic language.

Here is a strategic roadmap of improvements, ordered from the easiest (quick configuration tweaks) to the most complex (architectural overhauls).

---

### Level 1: Easiest (Configuration & Parameter Tuning)
*These require minimal code changes and will yield immediate improvements in speed and accuracy.*

**1. Switch to an Arabic/Multilingual Reranker (Crucial)**
* **The Issue:** You are using `BAAI/bge-reranker-base`, which is primarily optimized for English and Chinese. In your tests, it is the biggest bottleneck (`reranker_time` is 1.7s - 2.8s).
* **The Fix:** Switch to **`BAAI/bge-reranker-v2-m3`** or **`jinaai/jina-reranker-v2-base-multilingual`**. These models are explicitly trained on multilingual data (including Arabic) and will drastically improve retrieval accuracy for Arabic administrative text while likely reducing latency.

**2. Reduce Initial Retrieval `k` Value**
* **The Issue:** You are fetching `k=5` for both sparse and dense (10 documents total) before sending them to the reranker. 
* **The Fix:** Drop the initial retrieval to **`k=3` or `k=4`**. This reduces the payload sent to the reranker, which will directly decrease your `reranker_time` without sacrificing the quality of the final top-3 documents.

**3. Optimize Query Expansion**
* **The Issue:** Query expansion is adding 0.2s to 0.7s (`expansion_time`) using `llama-3.1-8b-instant`. 
* **The Fix:** Make it **conditional**. Only trigger query expansion if the user's prompt is very short (e.g., < 5 words) or lacks context. Alternatively, use a much smaller, faster model (like **Llama 3.2 1B or 3B**) specifically for the expansion step to cut this time in half.

---

### Level 2: Medium (Pipeline & Retrieval Enhancements)
*These require moderate adjustments to your retrieval logic and caching mechanisms.*

**4. Change Hybrid Search Fusion Method**
* **The Issue:** You are using a linear weight (`0.6 sparse, 0.4 dense`). Arabic is a highly morphological language, meaning exact keyword matches (sparse) and semantic meaning (dense) often have very different score distributions.
* **The Fix:** Switch from linear weighting to **Reciprocal Rank Fusion (RRF)**. RRF is generally much more robust for hybrid search in non-English languages because it relies on the rank of the documents rather than their raw scores.

**5. Implement Semantic Caching**
* **The Issue:** Your logs show `"cached": "False"` for all queries. Standard Redis caching only caches *exact* string matches. If a user asks the same question with slightly different wording, it won't hit the cache.
* **The Fix:** Implement **Semantic Caching** (e.g., using Redis with vector similarity, or a library like GistCache). This will cache the *meaning* of the query, drastically reducing `total_time` and LLM token costs for repetitive administrative questions.

**6. Upgrade the Embedding Model**
* **The Issue:** `intfloat/multilingual-e5-large` is good, but newer models handle Arabic context much better.
* **The Fix:** Consider upgrading to **`jina-embeddings-v3`**, **`nomic-embed-text-v1.5`**, or an Arabic-specific model like **`sandbox-solutions/arabic-embeddings`**. This will improve the `retrieval_time` accuracy before the reranker even kicks in.

---

### Level 3: Most Complex (Infrastructure & Advanced Architecture)
*These require significant refactoring, infrastructure changes, or advanced AI techniques.*

**7. Migrate the Vector Database**
* **The Issue:** ChromaDB is excellent for prototyping, but it can struggle with high-concurrency hybrid search and complex metadata filtering in production.
* **The Fix:** Migrate to **Qdrant** or **Milvus**. Both have native, highly optimized, and lightning-fast support for hybrid search (sparse + dense vectors) and will easily handle the scale of a national administrative chatbot.

**8. Implement Context Compression**
* **The Issue:** Your `llm_time` and `token_usage` can be reduced. Passing 10 raw chunks to Llama 3.3 70B is inefficient.
* **The Fix:** Introduce a **Context Compressor** (like LLMLingua) or an LLM-based extraction step. This step will read the retrieved chunks and strip out irrelevant text, passing only the highly dense, relevant Arabic context to the final LLM. This reduces prompt tokens and speeds up generation.

**9. Agentic RAG / Self-RAG**
* **The Issue:** Your current pipeline is linear (Expand -> Retrieve -> Rerank -> Generate). If the retrieval fails, the LLM hallucinates.
* **The Fix:** Implement an **Agentic Workflow** (using LangGraph or LlamaIndex). The agent can evaluate the retrieved context, decide if it's sufficient to answer the question, and if not, automatically rephrase the query and retrieve again. It can also route queries to different tools (e.g., SQL database for structured citizen data vs. Vector DB for policy documents).

**10. Domain-Specific Fine-Tuning**
* **The Issue:** The test questions are highly specific to Moroccan administrative procedures (e.g., law 19-55, `idarati.ma`). General LLMs might miss local nuances or Darija/MSA mixing.
* **The Fix:** Fine-tune a smaller, highly efficient model (like **Llama 3.1 8B** or **Qwen 2.5 7B**) specifically on Moroccan administrative documents, legal texts, and local dialects. This allows you to eventually drop the heavy 70B model, reducing latency and costs while maximizing domain accuracy.
