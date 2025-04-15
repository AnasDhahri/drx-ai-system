import json
import faiss
import numpy as np
import pickle
import os


def save_index(index, metadata, folder="faiss_index"):
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "index.faiss"))
    with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def load_index(folder="faiss_index"):
    index = faiss.read_index(os.path.join(folder, "index.faiss"))

    with open(os.path.join(folder, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def build_faiss_index(embedded_data, dims=768):
    """
    embedded_data: list of dicts with keys 'embedding', 'text', 'metadata'
    """
    embeddings = []
    meta = []

    for item in embedded_data:
        try:
            vec = np.array(item["embedding"], dtype=np.float32)
            if vec.ndim == 1 and vec.shape[0] == dims:
                embeddings.append(vec)
                meta.append({
                    "text": item["text"],
                    "metadata": item["metadata"]
                })
        except Exception as e:
            print(f" skipping invalid embedding: {e}")

    embeddings_np = np.stack(embeddings)
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings_np)

    return index, meta


def search(query_embedding_np, index, metadata, k=5):
    D, I = index.search(query_embedding_np, k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            result = metadata[idx]
            results.append({
                "text": result["text"],
                "metadata": {
                    "filename": result.get("filename", ""),
                    "new_chunk_id": result.get("new_chunk_id", "")
                }
            })
    return results



if __name__ == "__main__":
    from embedder import embed_chunks
    from chunker import split_into_chunks
    from extractor import load_all_files

    print("\n🔧 building vector store...")
    docs = load_all_files("data")
    chunks = split_into_chunks(docs, max_tokens=500)

    embedded = embed_chunks(chunks)

    #  filter out invalid embeddings
    embedded = [
        e for e in embedded
        if isinstance(e["embedding"], (list, np.ndarray)) and all(isinstance(x, (float, int)) for x in e["embedding"])
    ]

    if not embedded:
        print(" No valid embeddings found. Check upstream processing or data.")
        exit(1)

    dims = len(embedded[0]["embedding"])
    index, metadata = build_faiss_index(embedded, dims=dims)
    save_index(index, metadata)

    print(f"\n faiss index built and saved with {len(embedded)} vectors.")


