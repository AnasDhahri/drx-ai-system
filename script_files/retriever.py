import numpy as np
import re
import argparse
import json
import os
from embedder import embed_chunks
from chunker import split_into_chunks
from extractor import load_all_files
from vector_store import load_index, search
from translate import translate_text  # make sure you have this module


def is_readable(text):
    if len(text.strip()) < 30:
        return False
    if re.fullmatch(r"[^\w]*", text):  # all symbols or empty
        return False
    if len(re.findall(r'\d', text)) > len(text) * 0.6:  # mostly numbers
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Retriever with Translation and Summary Support")
    parser.add_argument("--lang", type=str, default="ar", help="Target language code (e.g. 'ar', 'fr', 'es')")
    parser.add_argument("--summary", action="store_true", help="Show summaries when available")
    args = parser.parse_args()
    target_lang = args.lang

    print(f"\U0001F7E2 retriever running. type your questions below (type 'exit' to quit)\n")

    index, metadata = load_index()

    # load summaries if requested
    summaries = {}
    if args.summary:
        try:
            with open(os.path.join("summaries", "summaries.json"), "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                summaries = {s["filename"]: s["summary"] for s in summary_data}
        except:
            summaries = {}

    while True:
        query = input(" your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        query_embedding = embed_chunks([{"text": query, "filename": "", "type": "", "chunk_id": "", "new_chunk_id": ""}])[0]["embedding"]
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        results = search(query_embedding_np, index, metadata)

        print("\n top results:\n")
        shown = 0
        for i, r in enumerate(results):
            if is_readable(r["text"]):
                shown += 1
                print(f"{shown}. [{r['metadata']['filename']} > {r['metadata']['new_chunk_id']}]\n{r['text'][:500]}...\n")
            if shown >= 5:
                break

        best_match = results[0]["text"] if results else "No good match found."
        translated = translate_text(best_match, target_lang)

        print("\n final answer (EN):", best_match)
        print(f" translated ({target_lang.upper()}):", translated)

        if args.summary:
            filename = results[0]["metadata"]["filename"] if results else None
            summary = summaries.get(filename)
            if summary:
                print("\n summary preview:", summary)
