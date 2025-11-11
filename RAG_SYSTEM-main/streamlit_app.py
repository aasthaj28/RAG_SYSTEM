"""
RAG System - Streamlit Cloud Compatible
Simplified version with minimal dependencies
"""
# ---- Ensure src folder import path ----
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# --------------------------------------

import streamlit as st
import requests
from pathlib import Path
import time
import sys

st.set_page_config(page_title="RAG System - PDF Q&A", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: bold; color: #1E88E5; text-align: center; }
    .answer-box { background: rgba(30,136,229,.08); padding: 1rem; border-radius: .5rem; }
    .source-box { background: rgba(30,136,229,.05); padding: .8rem; border-left: 4px solid #1E88E5; margin-bottom:.5rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})
        return model, collection
    except:
        return None, None

def query_huggingface(prompt, hf_token=None):
    models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct"
    ]
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.4}}

    for model in models:
        try:
            resp = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload)
            if resp.status_code == 200 and isinstance(resp.json(), list):
                return resp.json()[0].get("generated_text", "").strip()
        except:
            pass
    return "⚠️ No model available. Add HuggingFace token for priority."

def process_pdf(file_path):
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except: pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(open(file_path, 'rb'))
        return "".join(page.extract_text() or "" for page in reader.pages)
    except: pass
    return ""

def chunk_text(text, chunk_size=500, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# Session State
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.hf_token = ""
    st.session_state.chat_history = []

if not st.session_state.initialized:
    st.session_state.model, st.session_state.collection = load_models()
    st.session_state.initialized = True

st.markdown('<div class="main-header">🤖 RAG System</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 System Status")
    st.info("✅ HuggingFace token improves accuracy.")
    st.session_state.hf_token = st.text_input("HF Token:", type="password", value=st.session_state.hf_token)

    st.markdown("---")
    doc_count = st.session_state.collection.count() if st.session_state.collection else 0
    st.metric("Documents", doc_count)

    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf', 'txt'])

    if uploaded_file and st.button("🚀 Process"):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        text = process_pdf(tmp_path)
        chunks = chunk_text(text)
        embeds = st.session_state.model.encode(chunks)
        st.session_state.collection.add(
            documents=chunks,
            embeddings=embeds.tolist(),
            metadatas=[{'source': uploaded_file.name}] * len(chunks),
            ids=[f"id_{time.time()}_{i}" for i in range(len(chunks))]
        )
        os.unlink(tmp_path)
        st.success(f"✅ Added {len(chunks)} chunks!")

st.markdown("### 💬 Ask Questions")
question = st.text_input("Your question:")

if st.button("🚀 Ask"):
    query_emb = st.session_state.model.encode([question])
    results = st.session_state.collection.query(query_embeddings=query_emb.tolist(), n_results=3)
    context = "\n\n".join(results['documents'][0])

    prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the context.

Context:
{context}

Question:
{question}

Guidelines:
- Answer in 1–4 concise sentences
- Summarize instead of copying text
- Do NOT include phone numbers / emails
- If answer is not in context say: "I couldn't find that information in the documents."

Answer:
"""
    answer = query_huggingface(prompt, st.session_state.hf_token)

    st.session_state.chat_history.append((question, answer, results['documents'][0][:2]))

# Chat Display
for q, a, srcs in reversed(st.session_state.chat_history):
    st.markdown(f"**🙋 Q:** {q}")
    st.markdown(f'<div class="answer-box">{a}</div>', unsafe_allow_html=True)
    with st.expander(f"📚 Sources ({len(srcs)})"):
        for src in srcs:
            st.markdown(f'<div class="source-box">{src[:300]}...</div>', unsafe_allow_html=True)

st.markdown("---")
