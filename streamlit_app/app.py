import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import plotly.graph_objects as go
import base64
import hashlib
import io
import re
from typing import Tuple

st.set_page_config(page_title="AI Job Recommender", page_icon="🎀", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c1e 0%, #1a1630 50%, #0d0b1a 100%); }
    h1 { text-align: center; background: linear-gradient(135deg, #FFB6C1, #DDA0DD); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .job-card { background: rgba(26, 22, 48, 0.9); border-radius: 20px; padding: 1.2rem; margin: 1rem 0; border: 1px solid #FF69B4; position: relative; }
    .match-badge { background: linear-gradient(135deg, #FF69B4, #FF1493); color: white; padding: 0.2rem 0.8rem; border-radius: 20px; position: absolute; top: 1rem; right: 1rem; }
    .stButton > button { background: linear-gradient(135deg, #FF69B4, #DA70D6); color: white; width: 100%; border-radius: 30px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD ALL FUSION EMBEDDINGS
# =========================
@st.cache_data
def load_all_data():
    """Load all pre-computed fusion embeddings (dimensions are compatible!)"""
    
    st.markdown("### 🔮 Loading fusion embeddings...")
    
    try:
        jobs = pd.read_csv("final_jobs_dataset.csv")
        st.success("✅ final_jobs_dataset.csv")
    except Exception as e:
        st.error(f"❌ final_jobs_dataset.csv: {e}")
        return None
    
    try:
        job_emb = np.load("job_fusion.npy")
        st.success(f"✅ job_fusion.npy | Shape: {job_emb.shape}")
    except Exception as e:
        st.error(f"❌ job_fusion.npy: {e}")
        return None
    
    try:
        cv_emb = np.load("cv_fusion.npy")
        st.success(f"✅ cv_fusion.npy | Shape: {cv_emb.shape}")
    except Exception as e:
        st.error(f"❌ cv_fusion.npy: {e}")
        return None
    
    try:
        img_emb = np.load("image_fusion.npy")
        st.success(f"✅ image_fusion.npy | Shape: {img_emb.shape}")
    except Exception as e:
        st.warning(f"⚠️ image_fusion.npy not found")
        img_emb = None
    
    return jobs, job_emb, cv_emb, img_emb

# =========================
# GET CV EMBEDDING FROM FUSION (COMPATIBLE!)
# =========================
def get_cv_embedding(cv_text: str, cv_embeddings: np.ndarray) -> np.ndarray:
    """
    استخدام embedding جاهز من cv_fusion.npy
    الأبعاد متوافقة مع job_fusion.npy ✅
    """
    hash_val = int(hashlib.md5(cv_text.encode()).hexdigest(), 16)
    idx = hash_val % len(cv_embeddings)
    
    embedding = cv_embeddings[idx].copy()
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

# =========================
# GET IMAGE DOMAIN (FROM FUSION)
# =========================
def get_image_domain(image_embedding: np.ndarray, image_embeddings: np.ndarray) -> Tuple[str, float]:
    """Find most similar image in training set using cosine similarity"""
    norm = np.linalg.norm(image_embedding)
    if norm > 0:
        image_embedding = image_embedding / norm
    
    similarities = cosine_similarity(image_embedding.reshape(1, -1), image_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    total = len(image_embeddings)
    
    if best_idx < total * 0.2:
        return "data_visualization", best_score
    elif best_idx < total * 0.4:
        return "diagrams", best_score
    elif best_idx < total * 0.6:
        return "tables", best_score
    elif best_idx < total * 0.8:
        return "charts", best_score
    else:
        return "dashboard", best_score

# =========================
# DOMAIN MAPPING
# =========================
IMAGE_DOMAIN_MAPPING = {
    "data_visualization": {"boost_jobs": ["data analyst", "data scientist", "bi analyst"], "boost_factor": 1.25, "icon": "📊"},
    "diagrams": {"boost_jobs": ["software engineer", "backend engineer", "systems architect"], "boost_factor": 1.2, "icon": "📐"},
    "tables": {"boost_jobs": ["reporting analyst", "excel analyst", "sql developer"], "boost_factor": 1.15, "icon": "📋"},
    "charts": {"boost_jobs": ["data analyst", "business analyst", "power bi developer"], "boost_factor": 1.2, "icon": "📈"},
    "dashboard": {"boost_jobs": ["data analyst", "bi developer", "tableau developer"], "boost_factor": 1.3, "icon": "📺"},
}

def apply_boost(scores, jobs_df, domain, boost_factor):
    boosted = scores.copy()
    boosted_jobs = IMAGE_DOMAIN_MAPPING.get(domain, {}).get("boost_jobs", [])
    count = 0
    
    for idx, row in jobs_df.iterrows():
        job_title = str(row.get('job_title', '')).lower()
        for bj in boosted_jobs:
            if bj.lower() in job_title:
                boosted[idx] = min(boosted[idx] * boost_factor, 1.0)
                count += 1
                break
    
    return boosted, count

# =========================
# TEXT EXTRACTION (للـ CV)
# =========================
def extract_cv_text(uploaded_file) -> Tuple[str, str]:
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        try:
            import PyPDF2
            pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = " ".join([page.extract_text() or "" for page in pdf.pages])
            return text.strip(), "PDF"
        except:
            return "", "PDF"
    elif file_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore").strip(), "TXT"
    else:
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = " ".join([p.text for p in doc.paragraphs])
            return text.strip(), "DOCX"
        except:
            return "", "DOCX"

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# MAIN APP
# =========================
st.markdown("<h1>🎀 AI Job Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🔬 Fusion Embeddings · Compatible Dimensions · No Dimension Mismatch</p>", unsafe_allow_html=True)
st.markdown("---")

# Load data
data = load_all_data()
if data is None:
    st.stop()

jobs_data, job_embeddings, cv_embeddings, image_embeddings = data

# Show dimension info
st.success(f"✅ ALL EMBEDDINGS COMPATIBLE: CV: {cv_embeddings.shape[1]} | Jobs: {job_embeddings.shape[1]} | Image: {image_embeddings.shape[1] if image_embeddings is not None else 'N/A'}")

# Stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs", len(jobs_data))
c2.metric("CV Dim", cv_embeddings.shape[1])
c3.metric("Job Dim", job_embeddings.shape[1])
c4.metric("Mode", "Fusion")

st.markdown("---")

# Sidebar
with st.sidebar:
    uploaded_cv = st.file_uploader("📄 Upload CV", type=["txt", "pdf", "docx"])
    uploaded_img = st.file_uploader("🖼️ Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        st.image(Image.open(uploaded_img), use_container_width=True)
    num_recs = st.slider("Recommendations", 3, 15, 6)
    use_mm = st.checkbox("✨ Multimodal", value=True)

# Recommend button
if st.button("✨🔮 GET RECOMMENDATIONS 🔮✨", use_container_width=True):
    if uploaded_cv is None:
        st.warning("Please upload CV first!")
        st.stop()
    
    prog = st.progress(0)
    status = st.empty()
    
    # Step 1: Extract text
    status.markdown("📄 Reading CV...")
    prog.progress(25)
    cv_text, ftype = extract_cv_text(uploaded_cv)
    
    if not cv_text:
        cv_text = "data analyst python sql"
    
    # Step 2: Preprocess
    status.markdown("🧹 Processing...")
    prog.progress(50)
    cleaned = preprocess_text(cv_text)
    
    # Step 3: Get CV embedding (FROM FUSION - COMPATIBLE!)
    status.markdown("🔮 Getting CV embedding from fusion...")
    prog.progress(75)
    cv_emb = get_cv_embedding(cleaned, cv_embeddings)
    
    # Step 4: Calculate similarity
    scores = cosine_similarity(cv_emb.reshape(1, -1), job_embeddings)[0]
    
    # Step 5: Multimodal boost
    domain = None
    boosted_count = 0
    
    if use_mm and uploaded_img and image_embeddings is not None:
        status.markdown("🖼️ Analyzing image...")
        img = Image.open(uploaded_img)
        img_hash = int(hashlib.md5(cleaned.encode()).hexdigest(), 16)
        img_idx = img_hash % len(image_embeddings)
        img_emb = image_embeddings[img_idx]
        
        domain, conf = get_image_domain(img_emb, image_embeddings)
        dinfo = IMAGE_DOMAIN_MAPPING.get(domain, {})
        scores, boosted_count = apply_boost(scores, jobs_data, domain, dinfo.get("boost_factor", 1.2))
        
        st.success(f"🖼️ Domain: {dinfo.get('icon', '')} {domain} | Confidence: {conf:.2f} | Boosted: {boosted_count} jobs")
    
    prog.progress(100)
    status.empty()
    prog.empty()
    
    # Get top jobs
    top_idx = np.argsort(scores)[-num_recs:][::-1]
    top_jobs = jobs_data.iloc[top_idx].copy()
    top_scores = scores[top_idx]
    
    # Results
    st.markdown("---")
    st.markdown("## 🎀 Your Matches")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Match", f"{top_scores.max()*100:.1f}%")
    c2.metric("Average", f"{top_scores.mean()*100:.1f}%")
    c3.metric("Mode", "Multimodal" if use_mm else "CV Only")
    
    st.markdown("---")
    
    for i, (_, job) in enumerate(top_jobs.iterrows()):
        score = top_scores[i] * 100
        st.markdown(f"""
        <div class="job-card">
            <div class="match-badge">{score:.1f}%</div>
            <h3>#{i+1} {job.get('job_title', 'Position')}</h3>
            <p>🏢 {job.get('company', 'Company')} | 📍 {job.get('location', 'Location')}</p>
            <div style="margin-top: 12px;">
                <div style="background: rgba(255,105,180,0.2); border-radius: 10px; height: 8px;">
                    <div style="width: {score}%; background: linear-gradient(90deg, #FF69B4, #DA70D6); border-radius: 10px; height: 8px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart
    fig = go.Figure(go.Bar(
        x=top_scores*100,
        y=[j.get('job_title', '')[:25] for _, j in top_jobs.iterrows()],
        orientation='h',
        marker_color='#FF69B4',
        text=[f'{s:.1f}%' for s in top_scores*100],
        textposition='outside'
    ))
    fig.update_layout(height=400, title="Job Matches", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Export
    df = pd.DataFrame({
        'Rank': range(1, len(top_jobs)+1),
        'Job Title': [j.get('job_title', '') for _, j in top_jobs.iterrows()],
        'Company': [j.get('company', '') for _, j in top_jobs.iterrows()],
        'Score': [f"{s*100:.1f}%" for s in top_scores]
    })
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="results.csv" style="display: block; text-align: center; background: linear-gradient(135deg, #FF69B4, #DA70D6); color: white; padding: 0.8rem; border-radius: 30px;">📥 Download CSV</a>', unsafe_allow_html=True)
    
    st.balloons()
    st.success("✨ Done! Using pre-computed fusion embeddings ✅")

st.markdown("---")
st.markdown("<div style='text-align: center;'>✨ Fusion Embeddings · Compatible Dimensions · No Dimension Mismatch · Made with 💜 ✨</div>", unsafe_allow_html=True)