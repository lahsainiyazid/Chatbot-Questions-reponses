 **Cheat Sheet**. 
Here is your Redis Workflow and Commands Guide for your RAG API.

***

# Redis Caching Workflow & Commands for RAG API

## 1. The Caching Workflow
When a user sends a question to your FastAPI `/ask` endpoint, the following sequence occurs:

1. **Normalize & Hash:** The user's question is stripped of extra spaces, converted to lowercase, and hashed using MD5 to create a safe, fixed-length Redis key (e.g., `rag_cache:a1b2c3d4...`).
2. **Cache Lookup (`GET`):** The API checks Redis for this key.
   * **Cache Hit:** If found, Redis returns the cached JSON. The API skips the LLM and returns the response instantly.
   * **Cache Miss:** If not found (or if Redis is down), the API proceeds to the RAG pipeline.
3. **RAG Execution:** The question goes through Query Expansion $\rightarrow$ Hybrid Retrieval $\rightarrow$ Reranking $\rightarrow$ Final LLM Answer.
4. **Cache Save (`SETEX`):** The final response is saved back into Redis with a Time-To-Live (TTL) of `86400` seconds (24 hours).
5. **Return:** The API returns the response to the user.

---

## 2. Python Redis Commands Used in Your Code

These are the core commands your Python script uses to interact with Redis:

| Command | Python Syntax | Description |
| :--- | :--- | :--- |
| **Connect** | `redis.Redis(host, port, db, decode_responses=True)` | Initializes the connection. `decode_responses=True` ensures Redis returns standard strings instead of bytes. |
| **Read** | `redis_client.get(cache_key)` | Fetches the value for the key. Returns `None` if the key doesn't exist. |
| **Write + TTL**| `redis_client.setex(cache_key, 86400, json_data)` | Saves the JSON string to the key and sets an expiration time (TTL) of 86,400 seconds (24 hours). |

---

## 3. Redis CLI Commands (For Terminal Debugging)

You can use the `redis-cli` tool in your terminal to monitor and manage your cache. 

### Basic Connection & Health
```bash
# Test if Redis is running
redis-cli ping
# Expected output: PONG
```

### Inspecting the Cache
```bash
# Check if a specific question is cached (replace <hash> with actual MD5 hash)
redis-cli GET "rag_cache:<hash>"

# Check how much time is left before a key expires (TTL)
redis-cli TTL "rag_cache:<hash>"
# Returns: seconds remaining, or -2 if key doesn't exist, -1 if no expiry.

# Count how many RAG cache keys currently exist
redis-cli DBSIZE

# List all RAG cache keys (Warning: Do not use in production with millions of keys)
redis-cli KEYS "rag_cache:*"
```

### Managing Memory & Clearing Cache
```bash
# Check memory usage of the Redis database
redis-cli INFO memory | grep used_memory_human

# Delete one specific cached question
redis-cli DEL "rag_cache:<hash>"

# Clear ALL cached RAG questions (Use with caution!)
redis-cli KEYS "rag_cache:*" | xargs redis-cli DEL
```

---

## 4. Configuration & Best Practices Checklist

* [x] **Graceful Degradation:** Your code uses `try/except` blocks. If Redis crashes, your API will just run slightly slower without crashing.
* [x] **TTL (Time to Live):** Set to `86400` (24 hours). This prevents stale administrative data from staying in the cache forever.
* [x] **Memory Limits:** Ensure your Redis server has a `maxmemory` limit and an eviction policy (like `allkeys-lru`) set in `redis.conf` so it doesn't crash your server if the cache gets too full.
* [ ] **Security (Optional):** If your Redis server is exposed to the internet, ensure you add a `password` to your `redis.Redis(...)` connection in Python and require authentication in `redis.conf`.

***

*End of Document. To save this as a PDF, use your browser's Print function (Ctrl+P / Cmd+P) and select "Save as PDF".*
