import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(
    page_title="அறிஞர் அண்ணா கடிதங்கள் | Arignar Anna Archive",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Red & Black Theme
st.markdown("""
<style>
    /* Dark Theme Container Overrides */
    .stApp {
        background-color: #0f0f0f;
        color: #f5f5f5;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-left: 5px solid #dc2626;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    
    /* Badges */
    .badge-match {
        background-color: #991b1b;
        color: #fef2f2;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        border: 1px solid #ef4444;
    }
    
    .badge-meta {
        color: #a3a3a3;
        font-size: 0.85rem;
    }
    
    /* Excerpt Box */
    .excerpt-box {
        background-color: #121212;
        border: 1px solid #262626;
        border-radius: 6px;
        padding: 14px;
        margin: 12px 0;
        color: #e5e5e5;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Custom Links */
    a {
        color: #f87171 !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Engine Backend
@st.cache_resource
def load_backend():
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    index = faiss.read_index("anna_letters_faiss.index")
    with open("chunks_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

with st.spinner("Initializing AI Search Engine..."):
    model, index, metadata = load_backend()

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("<h2 style='color: #dc2626;'>🔴📜 Controls</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    top_k = st.slider("Max Search Results", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.metric(label="Total Indexed Chunks", value=len(metadata))
    st.caption("AI Model: `multilingual-e5-large`")

# 5. Header / Hero Section
st.markdown("<h1 style='color: #ffffff; margin-bottom: 0;'>📜 அறிஞர் அண்ணா கடிதக் களஞ்சியம்</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #dc2626; font-size: 1.1rem; font-weight: 600;'>Arignar Anna Historical Letter Archive</p>", unsafe_allow_html=True)
st.caption("Search letters by topic or political context in Tamil or English.")

st.markdown("---")

# 6. Quick Search Topic Buttons
st.write("##### 🔍 Frequent Topics:")
col1, col2, col3, col4 = st.columns(4)

search_topic = ""
if col1.button("மாநில சுயாட்சி", use_container_width=True):
    search_topic = "மாநில சுயாட்சி"
if col2.button("இந்தி எதிர்ப்பு", use_container_width=True):
    search_topic = "இந்தி எதிர்ப்பு போராட்டம்"
if col3.button("State Autonomy", use_container_width=True):
    search_topic = "State Autonomy and Rights"
if col4.button("Self Respect", use_container_width=True):
    search_topic = "Self Respect Movement"

# 7. Search Input
user_query = st.text_input(
    "Enter query:", 
    value=search_topic if search_topic else "", 
    placeholder="e.g., மாநில சுயாட்சி, இந்தி எதிர்ப்பு, Language Rights...",
    label_visibility="collapsed"
)

# 8. Vector Search Execution & Card Rendering
if user_query:
    query_vector = model.encode([f"query: {user_query}"], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector).astype('float32'), k=top_k)
    
    st.markdown(f"### Search Results for: <span style='color: #dc2626;'>\"{user_query}\"</span>", unsafe_allow_html=True)
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        item = metadata[idx]
        match_pct = f"{score * 100:.1f}%"
        
        # Red & Black Theme Result Card
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; font-size: 1.2rem; color: #ffffff;">📖 #{rank} {item['title']}</h3>
                    <span class="badge-match">{match_pct} Match</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span class="badge-meta">📰 Source: {item.get('source', 'General')}</span>
                    <span class="badge-meta" style="margin-left: 12px;">📅 Date: {item.get('date', 'N/A')}</span>
                </div>
                <div class="excerpt-box">
                    "{item['chunk_text']}"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'url' in item and item['url']:
                st.markdown(f"[🔗 Read full original letter on website]({item['url']})")
                
            st.write("") # Spacing
