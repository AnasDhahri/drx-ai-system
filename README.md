This system is a lightweight AI-powered document processing pipeline capable of:

Ingesting and chunking mixed-format files (PDF, DOCX, XLSX).

Creating a local vector database using faiss with Nomic embeddings.

Performing document retrieval via RAG-style question answering.

Translating answers into any language using Google Translate.

Summarizing each document using a local HuggingFace summarizer model.

Running edge case validation and automated test workflows.

 Folder Structure
bash
Copy
Edit
├── data/               # Place your documents (PDF/DOCX/XLSX) here
├── faiss_index/        # Auto-generated vector index + metadata
├── summaries/          # Auto-generated summaries
├── scripts/            # All core scripts (.py files)
│   ├── build_index.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── extractor.py
│   ├── retriever.py
│   ├── summarizer.py
│   ├── test_all.py
│   └── translate.py
└── README.txt          # You are here
 Setup Instructions
Create Environment

bash
Copy
Edit
python -m venv venv
Activate Environment

bash
Copy
Edit
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Unix/macOS
Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install faiss-cpu transformers pymupdf openpyxl python-docx pandas nomic googletrans tqdm
 How to Run
Step 1: Build the Index
bash
Copy
Edit
python scripts/build_index.py
Step 2: Run Retriever
bash
Copy
Edit
python scripts/retriever.py --lang ar
You can replace ar with any language code like fr, es, de, etc.

You can also enable summaries:

bash
Copy
Edit
python scripts/retriever.py --lang ar --summary
Step 3: Run Summarizer
bash
Copy
Edit
python scripts/summarizer.py
Results saved to:

bash
Copy
Edit
summaries/summaries.json
Step 4: Run All Tests Automatically
bash
Copy
Edit
python scripts/test_all.py
 Features
Works offline after model downloads

Automatically detects and summarizes multiple file types

Robust CLI interface

Summary preview and multilingual support

Modular architecture (easy to upgrade models or embeddings)

 Notes
Long documents are token-split for summarization.

All embeddings use nomic-embed-text-v1.

Summarizer uses sshleifer/distilbart-cnn-12-6.

Translations use googletrans.

