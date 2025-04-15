import tiktoken

# use the same tokenizer OpenAI models use (also good for LLaMA prompt fitting)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_into_chunks(docs, max_tokens=500, overlap=50):
    """
    Takes a list of dicts like from extractor.py and returns smaller token-limited chunks.
    Each returned dict has: filename, type, original_chunk_id, new_chunk_id, text
    """
    chunked_docs = []

    for doc in docs:
        text = doc["text"]
        tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            # no need to chunk
            chunked_docs.append({
                **doc,
                "new_chunk_id": f"{doc['chunk_id']}-0"
            })
        else:
            # slide window over tokens
            start = 0
            part = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)

                chunked_docs.append({
                    **doc,
                    "text": chunk_text,
                    "new_chunk_id": f"{doc['chunk_id']}-p{part}"
                })

                start += max_tokens - overlap
                part += 1

    return chunked_docs

# demo
if __name__ == "__main__":
    import json
    from extractor import load_all_files

    data_dir = "data"
    extracted_docs = load_all_files(data_dir)
    chunked = split_into_chunks(extracted_docs, max_tokens=500)

    print(f"\n produced {len(chunked)} token-sized chunks\n")
    for c in chunked[:3]:
        print(json.dumps(c, indent=2))
