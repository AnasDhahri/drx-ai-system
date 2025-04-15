import os
import json
from pathlib import Path
from extractor import load_all_files
from transformers import pipeline
from tqdm import tqdm
import torch


data_dir = str(Path(__file__).resolve().parent.parent / "data")
output_dir = str(Path(__file__).resolve().parent.parent / "summaries")
os.makedirs(output_dir, exist_ok=True)

print(f"📂 Loading files from: {data_dir}")
documents = load_all_files(data_dir)


device = 0 if torch.cuda.is_available() else -1
print("🤖 Loading local summarization model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)


results = []
print(f"📝 Summarizing {len(documents)} documents...")

for doc in tqdm(documents):
    try:
        # Split long text into 1024-token chunks
        text = doc["text"]
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = summarizer(chunks, max_length=130, min_length=30, do_sample=False)
        combined_summary = " ".join(s["summary_text"] for s in summaries)

        results.append({
            "filename": doc["filename"],
            "summary": combined_summary
        })

    except Exception as e:
        results.append({
            "filename": doc["filename"],
            "summary": f"[error: {str(e)}]"
        })


output_path = os.path.join(output_dir, "summaries.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f" Done! Summaries saved to: {output_path}")
print(" Preview:")
print(json.dumps(results[:1], indent=2, ensure_ascii=False))
