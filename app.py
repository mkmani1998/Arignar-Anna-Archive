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

# 2. Custom CSS for Modern Card Styling & Badges
st.markdown("""
<style>
    /* Card Container Styling */
    .result-card {
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    /* Badge styling */
    .badge-similarity {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    .badge-source {
        background-color: #f1f5f9;
        color: #475569;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    /* Quote Highlight */
    .quote-box {
        border-left: 4px solid #dc2626;
        padding-left: 14px;
        margin: 12px 0;
        font-size: 1.05rem;
        line-height: 1.6;
        font-style: italic;
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

# 4. Sidebar Controls & Filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a2/C.N.Annadurai.jpg", width=120)
    st.title("Archive Controls")
    
    st.markdown("---")
    top_k = st.slider("Max Search Results", min_value=1, max_value=10, value=5)
    
    # Filter by source magazine if present in metadata
    available_sources = sorted(list(set([m.get('source', 'General') for m in metadata if m.get('source')])))
    if available_sources:
        selected_sources = st.multiselect("Filter Source Magazine:", available_sources, default=available_sources)
    else:
        selected_sources = []
        
    st.markdown("---")
    st.metric(label="Total Indexed Chunks", value=len(metadata))
    st.caption("AI Model: `multilingual-e5-large`")

# 5. Header / Hero Banner
col_header, col_stats = st.columns([3, 1])

with col_header:
    st.title("📜 அறிஞர் அண்ணா கடிதக் களஞ்சியம்")
    st.markdown("##### *Arignar Anna Historical Letter Archive*")
    st.caption("Search letters by topic or political context in Tamil or English.")

with col_stats:
    st.info("💡 **Tip:** Semantic search understands context (e.g. searching *Language Agitation* finds *இந்தி எதிர்ப்பு*).")

st.markdown("---")

# 6. Interactive Quick Search Pills
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

# 7. Main Search Input
user_query = st.text_input(
    "Enter query:", 
    value=search_topic if search_topic else "", 
    placeholder="e.g., மாநில சுயாட்சி, இந்தி எதிர்ப்பு, Language Rights...",
    label_visibility="collapsed"
)

# 8. Search Execution & Card Results Rendering
if user_query:
    query_vector = model.encode([f"query: {user_query}"], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector).astype('float32'), k=top_k * 2) # Fetch extra to filter
    
    st.markdown(f"### Results for: *\"{user_query}\"*")
    
    results_found = 0
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        item = metadata[idx]
        
        # Apply Sidebar Source Filter
        item_source = item.get('source', 'General')
        if selected_sources and item_source not in selected_sources:
            continue
            
        results_found += 1
        match_pct = f"{score * 100:.1f}%"
        
        # Modern Card Container Layout
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; font-size: 1.25rem;">📖 {item['title']}</h3>
                    <span class="badge-similarity">{match_pct} Context Match</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="badge-source">📰 {item_source}</span>
                    <span style="font-size: 0.85rem; color: #64748b; margin-left: 10px;">📅 Date: {item.get('date', 'N/A')}</span>
                </div>
                <div class="quote-box">
                    "{item['chunk_text']}"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action button / link
            if 'url' in item and item['url']:
                st.markdown(f"[🔗 Read full original letter on website]({item['url']})")
            
            st.write("") # Spacing between cards
            
        if results_found >= top_k:
            break
            
    if results_found == 0:
        st.warning("No matching letters found for the selected filters.")
        
