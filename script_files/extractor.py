import os
import fitz
import docx
import pandas as pd
from pathlib import Path

def extract_pdf(file_path):
    texts = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                texts.append({
                    "filename": Path(file_path).name,
                    "type": "pdf",
                    "chunk_id": f"page-{page_num + 1}",
                    "text": text
                })
    except Exception as e:
        print(f"[PDF ERROR] {file_path}: {e}")
    return texts

def extract_docx(file_path):
    texts = []
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for i, para in enumerate(paragraphs):
            texts.append({
                "filename": Path(file_path).name,
                "type": "docx",
                "chunk_id": f"para-{i + 1}",
                "text": para
            })
    except Exception as e:
        print(f"[DOCX ERROR] {file_path}: {e}")
    return texts

def extract_excel(file_path):
    texts = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet).dropna(how='all').fillna('')
            for i, row in df.iterrows():
                text = ' | '.join([str(cell).strip() for cell in row if str(cell).strip()])
                if text:
                    texts.append({
                        "filename": Path(file_path).name,
                        "type": "xlsx",
                        "chunk_id": f"{sheet}-row-{i + 1}",
                        "text": text
                    })
    except Exception as e:
        print(f"[XLSX ERROR] {file_path}: {e}")
    return texts

def load_all_files(data_dir):
    all_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.lower().split('.')[-1]
            if ext == "pdf":
                all_data.extend(extract_pdf(file_path))
            elif ext in ["docx"]:
                all_data.extend(extract_docx(file_path))
            elif ext in ["xlsx", "xls"]:
                all_data.extend(extract_excel(file_path))
            else:
                print(f"[SKIPPED] Unsupported file: {file}")
    return all_data

if __name__ == "__main__":
    data_dir = "data"
    extracted = load_all_files(data_dir)
    print(f"\n extracted {len(extracted)} text chunks\n")
    # preview first 3 chunks
    for item in extracted[:3]:
        print(item)
