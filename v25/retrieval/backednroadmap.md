
### 🗺️ Roadmap: MongoDB & FastAPI CRUD for RAG

**Goal:** Create a robust API to manage the `chunks` collection in your `rag_db`.

1.  **Step 1: Pydantic Models & The `_id` Problem**
    *   Learn how to map MongoDB's BSON `_id` (ObjectId) to a Python string using Pydantic.
    *   Define models for creating and reading your document chunks.

2.  **Step 2: Database Connection Setup**
    *   Initialize `MongoClient` in FastAPI using environment variables (like you did in `rag.py`).
    *   Use FastAPI's `lifespan` or startup events to ensure a clean connection.

3.  **Step 3: CREATE (POST Endpoint)**
    *   Build an endpoint to insert new text chunks into MongoDB.
    *   Handle automatic ID generation and data validation.

4.  **Step 4: READ (GET Endpoints)**
    *   Fetch all items (with optional pagination).
    *   Fetch a single item by its unique ID.

5.  **Step 5: UPDATE (PATCH Endpoint)**
    *   Modify specific fields of an existing chunk (e.g., updating the `text` content).

6.  **Step 6: DELETE (DELETE Endpoint)**
    *   Remove a chunk from the database by its ID.

