import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="அறிஞர் அண்ணா கடிதங்கள் | Arignar Anna Archive",
    page_icon="📜",
    layout="wide"
)

@st.cache_resource
def load_backend():
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    index = faiss.read_index("anna_letters_faiss.index")
    with open("chunks_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

with st.spinner("Loading semantic search engine..."):
    model, index, metadata = load_backend()

st.title("📜 அறிஞர் அண்ணா கடிதக் களஞ்சியம்")
st.caption("AI-Powered Contextual Search Engine for Arignar Anna's Letters")

query = st.text_input("Enter Search Topic (Tamil or English):", placeholder="e.g., மாநில சுயாட்சி, இந்தி எதிர்ப்பு, Language Agitation...")

if query:
    query_vector = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector).astype('float32'), k=5)
    
    st.subheader(f"Search Results for: '{query}'")
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        item = metadata[idx]
        match_pct = f"{score * 100:.1f}%"
        
        with st.expander(f"#{rank} | 📖 {item['title']} (Match: {match_pct})", expanded=(rank == 1)):
            st.markdown(f"> \"{item['chunk_text']}\"")
            if 'url' in item and item['url']:
                st.markdown(f"[🔗 Read full letter on original site]({item['url']})")
