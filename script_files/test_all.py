import subprocess
import os
from pathlib import Path
import json

scripts_dir = Path(__file__).resolve().parent
summaries_path = scripts_dir.parent / "summaries" / "summaries.json"
venv_python = os.path.join(os.environ["VIRTUAL_ENV"], "Scripts", "python.exe")


print("\n step 1: rebuilding index")
try:
    subprocess.run([venv_python, "build_index.py"], cwd=scripts_dir, check=True)
except subprocess.CalledProcessError as e:
    print(f" build_index.py failed:\n{e}")
    exit(1)


print("\n step 2: testing summarizer")
try:
    subprocess.run([venv_python, "summarizer.py"], cwd=scripts_dir, check=True)
except subprocess.CalledProcessError as e:
    print(f" summarizer.py failed:\n{e}")
    exit(1)


print("\n checking output...")
if summaries_path.exists():
    try:
        with open(summaries_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
            print(f" found {len(summaries)} summaries.")
            print(" preview:\n")
            print(json.dumps(summaries[:1], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f" could not read summaries.json: {e}")
else:
    print(" summaries.json file not found.")

print("\n test_all.py completed.")
