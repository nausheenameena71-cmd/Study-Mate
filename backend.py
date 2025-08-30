import os
import io
import time
import random
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import numpy as np
import faiss

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Config
# ---------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ibm-granite/granite-embedding-30m-english")
GEN_MODEL       = os.getenv("GEN_MODEL",       "ibm-granite/granite-3-8b-instruct")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K           = int(os.getenv("TOP_K", "4"))

# Hugging Face token (required if models are gated)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="StudyMate Backend (IBM Granite + Hugging Face)")

# Allow your Streamlit app (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod (e.g., ["http://localhost:8501"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models (lazy init)
# ---------------------------
_embedding_model: Optional[SentenceTransformer] = None
_gen_pipe = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
    return _embedding_model

def get_gen_pipe():
    global _gen_pipe
    if _gen_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL,
            torch_dtype="auto",
            device_map="auto",
            use_auth_token=HF_TOKEN,
        )
        _gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )
    return _gen_pipe

# ---------------------------
# Simple in-memory “corpus”
# (swap to persistent store in prod)
# ---------------------------
class Corpus:
    def __init__(self):
        self.text: str = ""
        self.chunks: List[str] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.embeddings: Optional[np.ndarray] = None
        self.dim: Optional[int] = None

CORPUS = Corpus()

# ---------------------------
# Utils
# ---------------------------
def extract_text_from_pdf_bytes(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks: List[str]) -> None:
    model = get_embedding_model()
    embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    CORPUS.chunks = chunks
    CORPUS.embeddings = embs
    CORPUS.index = index
    CORPUS.dim = dim

def search_chunks(query: str, top_k=TOP_K) -> List[Dict[str, Any]]:
    if CORPUS.index is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    model = get_embedding_model()
    q = model.encode([query], convert_to_numpy=True)
    distances, indices = CORPUS.index.search(q, top_k)
    out = []
    for rank, (i, d) in enumerate(zip(indices[0], distances[0]), start=1):
        out.append({
            "rank": rank,
            "distance": float(d),
            "chunk": CORPUS.chunks[i],
            "index": int(i)
        })
    return out

def summarize_text(text: str, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    return (". ".join(sentences[:max_sentences]) + "...") if sentences else ""

def make_flashcards(text: str, num_cards: int = 5) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    cards = []
    for _ in range(min(num_cards, max(1, len(sentences)//2))):
        q = random.choice(sentences)
        a = random.choice(sentences)
        cards.append({"question": q, "answer": a})
    return cards

RAG_PROMPT = """You are StudyMate, a helpful academic assistant.
Answer the user's question using ONLY the provided context. If the answer
is not in the context, say you don't know.

Question:
{question}

Context:
{context}

Answer:"""

def answer_with_context(question: str, retrieved: List[Dict[str, Any]]) -> str:
    ctx = "\n\n---\n\n".join([r["chunk"] for r in retrieved])
    prompt = RAG_PROMPT.format(question=question, context=ctx)
    gen = get_gen_pipe()
    out = gen(prompt, eos_token_id=None)[0]["generated_text"]
    # Return only the part after "Answer:"
    if "Answer:" in out:
        return out.split("Answer:", 1)[-1].strip()
    return out.strip()

# ---------------------------
# Schemas
# ---------------------------
class IngestResponse(BaseModel):
    chunks: int
    dim: int
    message: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]

class SummaryResponse(BaseModel):
    summary: str

class FlashcardsResponse(BaseModel):
    cards: List[Dict[str, str]]

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    texts = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDFs supported, got {f.filename}")
        data = await f.read()
        try:
            txt = extract_text_from_pdf_bytes(data)
            texts.append(txt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse {f.filename}: {e}")

    CORPUS.text = "\n".join(texts)
    chunks = chunk_text(CORPUS.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDFs.")
    build_faiss_index(chunks)

    return IngestResponse(
        chunks=len(CORPUS.chunks),
        dim=int(CORPUS.dim or 0),
        message="Documents uploaded and indexed with IBM Granite embeddings."
    )

@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    retrieved = search_chunks(body.query, top_k=TOP_K)
    answer = answer_with_context(body.query, retrieved)
    return QueryResponse(query=body.query, answer=answer, sources=retrieved)

@app.get("/summary", response_model=SummaryResponse)
def summary():
    if not CORPUS.text:
        raise HTTPException(status_code=400, detail="No documents available. Upload first.")
    return SummaryResponse(summary=summarize_text(CORPUS.text))

@app.get("/flashcards", response_model=FlashcardsResponse)
def flashcards(n: int = 5):
    if not CORPUS.text:
        raise HTTPException(status_code=400, detail="No documents available. Upload first.")
    return FlashcardsResponse(cards=make_flashcards(CORPUS.text, num_cards=n))
