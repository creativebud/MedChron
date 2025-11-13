# rag_store.py
import os, uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# One persistent collection per workspace
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.getcwd(), "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_records")

def get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

def get_collection():
    client = get_client()
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return client.create_collection(COLLECTION)

def upsert_chunks(chunks: List[Dict[str, Any]]):
    """
    Upsert chunks to ChromaDB in batches to avoid memory issues.
    chunks: [{id?, text, metadata}]
    """
    if not chunks: 
        return
    
    coll = get_collection()
    from embeddings import embed_texts
    
    # Process in batches to avoid memory overflow
    BATCH_SIZE = 100  # Process 100 chunks at a time
    total_chunks = len(chunks)
    
    print(f"Upserting {total_chunks} chunks in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        ids = []
        docs = []
        metas = []
        
        for c in batch:
            ids.append(c.get("id") or str(uuid.uuid4()))
            docs.append(c["text"])
            metas.append(c.get("metadata", {}))
        
        # Embed the batch
        embs = embed_texts(docs)
        
        # Upsert the batch
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        
        if i + BATCH_SIZE < total_chunks:
            print(f"  Processed {i + len(batch)}/{total_chunks} chunks...")
    
    print(f"✓ Successfully upserted {total_chunks} chunks")

def search(query: str, k: int = 12) -> List[Dict[str, Any]]:
    from embeddings import embed_one
    coll = get_collection()
    q = embed_one(query)
    res = coll.query(query_embeddings=[q], n_results=k, include=["documents","metadatas","distances","embeddings"])
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return out
