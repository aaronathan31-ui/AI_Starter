"""Run with: streamlit run app.py"""
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.pipeline import SearchIndex, extract_pdf
from src.answer import answer_with_sources

load_dotenv()
st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.title("📚 DocuMind AI")
st.caption("Ask questions about PDFs with traceable source passages.")


@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("Upload text-based PDFs", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Sources per answer", 1, 6, 3)
    st.caption("Text-only extraction: image-only/scanned pages require OCR (not included).")
    st.caption("Files are handled in memory for this session; avoid uploading confidential documents to a public deployment.")

if not uploads:
    st.info("Upload one or more PDFs in the sidebar to begin.")
    st.stop()

signature = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in uploads)
if st.session_state.get("signature") != signature:
    all_chunks = []
    errors = []
    for f in uploads:
        if len(f.getvalue()) > 15 * 1024 * 1024:
            errors.append(f"{f.name}: Larger than 15 MB; skipped")
            continue
        try:
            all_chunks.extend(extract_pdf(f.getvalue(), f.name))
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
    for error in errors:
        st.warning(error)
    if not all_chunks:
        st.error("No searchable text found in those files.")
        st.stop()
    with st.spinner("Embedding document chunks (first run downloads the model)..."):
        st.session_state["index"] = SearchIndex(all_chunks, get_model())
    st.session_state["signature"] = signature
    st.session_state["messages"] = []

index = st.session_state["index"]
st.success(f"Indexed {len(index.chunks)} text chunks from your uploaded PDFs.")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("hits"):
            with st.expander("Sources and relevant excerpts"):
                for i, (chunk, score) in enumerate(message["hits"], 1):
                    st.markdown(f"**[S{i}] {chunk.source} — page {chunk.page}** (similarity {score:.2f})")
                    st.write(chunk.text)

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state["messages"].append({"role": "user", "text": question})
    hits = index.search(question, top_k=top_k)
    try:
        answer = answer_with_sources(question, hits)
    except Exception as exc:
        answer = f"Answer generation failed ({type(exc).__name__}). Retrieved passages are available below. Check API key, billing, and model configuration."
    st.session_state["messages"].append({"role": "assistant", "text": answer, "hits": hits})
    st.rerun()
