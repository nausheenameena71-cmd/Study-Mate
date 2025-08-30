import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

# ---------------------------
# Configurations
# ---------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

st.set_page_config(page_title="📘 StudyMate - AI Academic Assistant", layout="wide")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)
def add_bg_from_url(image_url):
    """Set background image from a URL in Streamlit app."""
    css = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Example usage
add_bg_from_url("https://static.vecteezy.com/system/resources/thumbnails/020/734/052/large/animated-studying-lo-fi-background-late-night-homework-2d-cartoon-character-animation-with-nighttime-cozy-bedroom-interior-on-background-4k-footage-with-alpha-channel-for-lofi-music-aesthetic-video.jpg")
model = load_model()

# ---------------------------
# Helper Functions
# ---------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=pdf_file.getbuffer(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def search_index(query, model, index, chunks, top_k=3):
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)
    return f"**Q:** {query}\n\n**A:** Based on the documents:\n{context[:400]}..."

def summarize_text(text, max_sentences=3):
    sentences = text.split(". ")
    return ". ".join(sentences[:max_sentences]) + "..."

def create_flashcards(text, num_cards=5):
    sentences = text.split(". ")
    cards = []
    for _ in range(min(num_cards, len(sentences)//2)):
        q = random.choice(sentences).strip()
        a = random.choice(sentences).strip()
        cards.append((q, a))
    return cards

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📘 StudyMate – AI-Powered Academic Assistant For Study Grids")
st.write("Upload PDFs, ask questions, get summaries, and practice with flashcards.")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    full_text = ""
    for file in uploaded_files:
        full_text += extract_text_from_pdf(file) + "\n"

    chunks = chunk_text(full_text)
    index, _ = build_faiss_index(chunks, model)

    st.success("✅ Documents uploaded and indexed!")

    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📝 Summaries", "🎴 Flashcards"])

    with tab1:
        query = st.text_input("Ask a question:")
        if query:
            retrieved_chunks = search_index(query, model, index, chunks)
            answer = generate_answer(retrieved_chunks, query)
            st.session_state.history.append({"q": query, "a": answer, "src": retrieved_chunks})

        for chat in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {chat['q']}")
            st.markdown(f"**A:** {chat['a']}")
            with st.expander("📑 Sources"):
                for s in chat["src"]:
                    st.markdown(s)

    with tab2:
        st.subheader("Summary of Uploaded Documents")
        st.write(summarize_text(full_text))

    with tab3:
        st.subheader("Flashcards")
        cards = create_flashcards(full_text, num_cards=5)
        for i, (q, a) in enumerate(cards, 1):
            with st.expander(f"❓ Card {i}: {q}"):
                st.write(f"✅ {a}")
