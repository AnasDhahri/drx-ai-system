from nomic import embed
from chunker import split_into_chunks
from extractor import load_all_files

def embed_chunks(docs, model="nomic-embed-text-v1"):
    results = []

    texts = [doc["text"] for doc in docs]

    raw_output = embed.text(
        texts=texts,
        model=model
    )

    embeddings = raw_output["embeddings"]  #  fix here
    #print(f" first 3 embeddings: {embeddings[:3]}")

    for doc, emb in zip(docs, embeddings):
        results.append({
            "embedding": emb,
            "text": doc["text"],
            "metadata": {
                "filename": doc["filename"],
                "type": doc["type"],
                "chunk_id": doc["chunk_id"],
                "new_chunk_id": doc["new_chunk_id"]
            }
        })

    return results


# Demo usage
if __name__ == "__main__":
    print("\n loading and chunking documents...")
    docs = load_all_files("data")
    chunked = split_into_chunks(docs, max_tokens=500)

    print(f" total chunks: {len(chunked)}")

    print("\n generating embeddings using nomic...")
    embedded = embed_chunks(chunked)

    print(f"\n generated {len(embedded)} embeddings")
    print("example:\n")
    from pprint import pprint
    pprint(embedded[0])
