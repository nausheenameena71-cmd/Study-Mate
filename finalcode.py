
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"  # change if deployed

# ---- Fixed PDF path (change this to your actual file path) ----
pdf_path = r"D:\Hackathon\DEVOPS UNIT-1 (1).pdf"
path = Path(pdf_path)

if not path.exists():
    raise FileNotFoundError(f"❌ File not found: {path}")

files = {"files": open(path, "rb")}

# 1. Upload and index PDFs
resp = requests.post(f"{BASE_URL}/ingest", files=files)
print("Ingest response:", resp.json())

# 2. Ask a question (no user input, you can preset one)
query = {"query": "Summarize the main concepts"}
resp = requests.post(f"{BASE_URL}/query", json=query)
print("\nQuery response:")
print("Answer:", resp.json()["answer"])
print("Sources:", resp.json()["sources"])

# 3. Get a summary
resp = requests.get(f"{BASE_URL}/summary")
print("\nSummary:", resp.json()["summary"])

# 4. Get flashcards
resp = requests.get(f"{BASE_URL}/flashcards", params={"n": 3})
print("\nFlashcards:")
for card in resp.json()["cards"]:
    print("Q:", card["question"])
    print("A:", card["answer"])
    print("---")
