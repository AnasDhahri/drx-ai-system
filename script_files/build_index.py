import os
import numpy as np
import faiss
import json
from pathlib import Path
from extractor import load_all_files
from chunker import split_into_chunks
from embedder import embed_chunks

# config
data_dir = str(Path(__file__).resolve().parent.parent / "data")
index_dir = "faiss_index"
os.makedirs(index_dir, exist_ok=True)

print("🔍 loading and chunking files...")
docs = load_all_files(data_dir)
chunked = split_into_chunks(docs)

#  safety check
valid_chunks = [c for c in chunked if "text" in c and "chunk_id" in c and "new_chunk_id" in c]
print(f" using {len(valid_chunks)} valid chunks")

print(" embedding chunks...")
embedded = embed_chunks(valid_chunks)

print(" saving index...")
dimension = len(embedded[0]["embedding"])
index = faiss.IndexFlatL2(dimension)

embeddings_np = np.array([item["embedding"] for item in embedded], dtype=np.float32)
index.add(embeddings_np)

faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

# save metadata for retrieval
with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump([
        {
            "filename": item["metadata"]["filename"],
            "new_chunk_id": item["metadata"]["new_chunk_id"],
            "text": item["text"]
        }
        for item in embedded
    ], f, indent=2, ensure_ascii=False)

print(" index and metadata saved.")
