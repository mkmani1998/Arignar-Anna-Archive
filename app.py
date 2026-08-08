import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(
    page_title="அறிஞர் அண்ணா கடிதங்கள் | Arignar Anna Archive",
    page_icon="📜",
    layout="wide"
)

# 2. Custom CSS Injection for Red & Black Theme
st.markdown("""
<style>
    /* Main Background & Text Colors */
    .stApp {
        background-color: #0d0d0d;
        color: #f5f5f5;
    }
    
    /* Header Typography */
    h1 {
        color: #e63946 !important;
        font-weight: 700 !important;
    }
    
    .stCaption {
        color: #a3a3a3 !important;
        font-size: 1rem !important;
    }
    
    /* Input Text Box Styling */
    .stTextInput input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #dc2626 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #ff4d4d !important;
        box-shadow: 0 0 8px rgba(220, 38, 38, 0.6) !important;
    }

    /* Expander Container Customization */
    .st-emotion-cache-1f3w014, div[data-testid="stExpander"] {
        background-color: #171717 !important;
        border: 1px solid #333333 !important;
        border-left: 5px solid #dc2626 !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
    }

    /* Expander Header Text */
    div[data-testid="stExpander"] summary span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Excerpt Blockquote Styling */
    blockquote {
        background-color: #121212 !important;
        border-left: 3px solid #b91c1c !important;
        color: #e5e5e5 !important;
        padding: 12px 16px !important;
        border-radius: 4px !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    /* Hyperlinks */
    a {
        color: #f87171 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    a:hover {
        text-decoration: underline !important;
        color: #ef4444 !important;
    }
    
    /* Loading Spinner Text */
    .stSpinner > div {
        color: #dc2626 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Backend Engine Loader
@st.cache_resource
def load_backend():
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    index = faiss.read_index("anna_letters_faiss.index")
    with open("chunks_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

with st.spinner("Loading semantic search engine..."):
    model, index, metadata = load_backend()

# 4. App Title & Subtitle
st.title("📜 அறிஞர் அண்ணா கடிதக் களஞ்சியம்")
st.caption("AI-Powered Contextual Search Engine for Arignar Anna's Letters")

# 5. Search Bar Input
query = st.text_input(
    "Enter Search Topic (Tamil or English):", 
    placeholder="e.g., மாநில சுயாட்சி, இந்தி எதிர்ப்பு, Language Agitation..."
)

# 6. Search Execution & Display
if query:
    query_vector = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector).astype('float32'), k=5)
    
    st.markdown(f"<h3 style='color: #f5f5f5;'>Search Results for: <span style='color: #dc2626;'>'{query}'</span></h3>", unsafe_allow_html=True)
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        item = metadata[idx]
        match_pct = f"{score * 100:.1f}%"
        
        with st.expander(f"#{rank} | 📖 {item['title']} (Match: {match_pct})", expanded=(rank == 1)):
            st.markdown(f"> \"{item['chunk_text']}\"")
            if 'url' in item and item['url']:
                st.markdown(f"[🔗 Read full letter on original site]({item['url']})")
