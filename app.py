import streamlit as st
import pandas as pd
import os
import time
import logging
import concurrent.futures
import gc  # Garbage Collection

# Import pipeline components
from src.parser import extract_text
from src.analyzer import clean_text, extract_metadata, analyze_candidate
from src.embedder import DocumentEmbedder
from src.ranker import rank_candidates, generate_explainable_dataframe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Streamlit Page config
st.set_page_config(
    page_title="Intelligent Candidate Discovery - Enterprise Security Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Glassmorphism, Outfit Google Font, color-coded badges, dark theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main App Customization */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant Dark Slate Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    
    /* Hackathon Winner Premium Header */
    .header-container {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.5) 0%, rgba(17, 24, 39, 0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 35px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    }
    .header-badge {
        background: linear-gradient(90deg, #dc2626 0%, #ea580c 100%);
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 50px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .header-title {
        background: linear-gradient(90deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.25rem;
        font-weight: 300;
    }
    
    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 6px 30px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Green Pill Badges (Matched Skills) */
    .badge-matched {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 50px;
        padding: 5px 14px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Dark Red Pill Badges (Gap Skills) */
    .badge-gap {
        display: inline-block;
        background: rgba(153, 27, 27, 0.2);
        color: #f87171;
        border: 1px solid rgba(153, 27, 27, 0.4);
        border-radius: 50px;
        padding: 5px 14px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Security Warning Badge (Padding Alert) */
    .badge-warning {
        display: inline-block;
        background: rgba(217, 119, 6, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(217, 119, 6, 0.35);
        border-radius: 50px;
        padding: 5px 14px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Diagnostics Tag styling */
    .badge-diagnostic {
        display: inline-block;
        background: rgba(245, 158, 11, 0.08);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-family: monospace;
        margin-right: 8px;
    }
    
    /* Run Pipeline Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #818cf8 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 1.15rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        width: 100%;
        letter-spacing: 0.5px;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
        background: linear-gradient(90deg, #93c5fd 0%, #4f46e5 100%);
        color: white;
    }
    
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(129, 140, 248, 0.25);
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title Block
st.title("TalentMatch AI: Enterprise Sourcing & Discovery Hub")
st.caption("Advanced Semantic Alignment Engine • Anonymized Multi-Channel Evaluation Pipeline")
st.markdown("<br>", unsafe_allow_html=True)

# Initialize session states
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = {}
if 'embedder' not in st.session_state:
    st.session_state.embedder = DocumentEmbedder(use_local_transformer=True)

# Sidebar Settings
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛡️ Enterprise Security Settings</div>', unsafe_allow_html=True)
    
    anonymize_view = st.toggle("Anonymized Sourcing View", value=True,
                               help="Hides candidate names and contact data from expanded profile views to ensure absolute technical objectivity.")
    
    st.markdown('<div class="sidebar-header">⚙️ System Configurations</div>', unsafe_allow_html=True)
    
    use_transformer = st.toggle("Enable Local Transformer Model", value=True,
                                help="Toggle local SentenceTransformer ('all-MiniLM-L6-v2'). If disabled, runs fallback TF-IDF vectorizer.")
    
    st.markdown('<div class="sidebar-header">⚡ Thread pool Ingest</div>', unsafe_allow_html=True)
    max_workers = st.slider("Max Concurrent CPU Worker Threads", min_value=2, max_value=16, value=8, step=2)

    # Auto-adjust embedder mode if toggle changed
    if use_transformer != st.session_state.embedder.use_local_transformer:
        st.session_state.embedder = DocumentEmbedder(use_local_transformer=use_transformer)

# Parallel single file parsing function
def process_file_async(res_file, jd_text):
    start_time = time.time()
    try:
        file_bytes = res_file.read()
        file_name = res_file.name
        
        # 1. Ingest text (Capped at 3 pages maximum to prevent infinite loop crashes)
        raw_text, parse_method = extract_text(file_bytes, file_name)
        
        # 2. Analyze Candidate (Anti-Cheat padding filter + Anonymization + Skill Gap extraction)
        cleaned_text, metadata, nlp_mode, padding_flag, flagged_keywords = analyze_candidate(raw_text, jd_text)
        
        elapsed = time.time() - start_time
        return {
            "success": True,
            "name": os.path.splitext(file_name)[0],
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "metadata": metadata,
            "parse_method": parse_method,
            "nlp_mode": nlp_mode,
            "padding_penalty_applied": padding_flag,
            "flagged_keywords": flagged_keywords,
            "elapsed_sec": elapsed,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "name": os.path.splitext(res_file.name)[0],
            "raw_text": "",
            "cleaned_text": "",
            "metadata": {"email": "N/A", "phone": "N/A", "skills": [], "experience": 0, "matched_skills": [], "missing_skills": []},
            "parse_method": "FAILED",
            "nlp_mode": "FAILED",
            "padding_penalty_applied": False,
            "flagged_keywords": [],
            "elapsed_sec": elapsed,
            "error": str(e)
        }

# UI Columns Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card"><h3>1. Define Job Requirements</h3>', unsafe_allow_html=True)
    jd_input_method = st.radio("Choose Input Type:", ["Paste Text Requirements", "Upload JD Document"], horizontal=True)
    
    jd_text = ""
    if jd_input_method == "Paste Text Requirements":
        jd_text = st.text_area(
            "Target Job Description:", 
            height=280,
            placeholder="Specify years of experience required and critical technology stacks..."
        )
    else:
        uploaded_jd = st.file_uploader("Ingest JD File (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if uploaded_jd:
            try:
                jd_bytes = uploaded_jd.read()
                jd_text, parse_method = extract_text(jd_bytes, uploaded_jd.name)
                st.success(f"JD parsed successfully using {parse_method}!")
                jd_text = st.text_area("Ingested Job Description content:", value=jd_text, height=200)
            except Exception as e:
                st.error(f"Failed parsing Job Description: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card"><h3>2. Upload Candidate Profiles</h3>', unsafe_allow_html=True)
    uploaded_resumes = st.file_uploader(
        "Ingest resumes (PDF/DOCX) - Support batch-upload (20+ files)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    if uploaded_resumes:
        st.info(f"📂 Selected {len(uploaded_resumes)} resume files.")
        with st.expander("Ingested file queue"):
            for f in uploaded_resumes:
                st.caption(f"📑 {f.name} | Size: {f.size / 1024:.1f} KB")
    else:
        st.info("Upload one or more candidate profiles to launch parsing.")
    st.markdown('</div>', unsafe_allow_html=True)

# Run Sourcing Engine Button
if st.button("👑 Execute Sourcing & Hybrid Matching Engine"):
    if not jd_text.strip():
        st.error("⚠️ Please specify a Job Description requirements schema.")
    elif not uploaded_resumes:
        st.error("⚠️ Upload candidate resumes to perform scoring.")
    else:
        pipeline_start = time.time()
        
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # 1. Process Job Description
            status_box.write("⚙️ Ingesting and extracting Job Description metadata...")
            progress_bar.progress(10)
            
            clean_jd_text, _ = clean_text(jd_text)
            jd_metadata = extract_metadata(jd_text)
            
            # 2. Parallel thread pool execution for all resumes to prevent UI freeze
            status_box.write(f"🚀 Processing {len(uploaded_resumes)} resumes asynchronously (max workers={max_workers})...")
            progress_bar.progress(30)
            
            processed_profiles = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_file_async, f, jd_text): f for f in uploaded_resumes}
                for future in concurrent.futures.as_completed(futures):
                    processed_profiles.append(future.result())
                    
            progress_bar.progress(65)
            
            # Filter successful profiles
            valid_profiles = [p for p in processed_profiles if p["success"]]
            failed_profiles = [p for p in processed_profiles if not p["success"]]
            
            if not valid_profiles:
                st.error("❌ All candidate document parsing attempts failed. Check logs.")
            else:
                status_box.write("✨ Computing semantic vector embeddings...")
                progress_bar.progress(85)
                
                # Assemble corpus: [JD] + [Resumes]
                corpus = [clean_jd_text] + [p["cleaned_text"] for p in valid_profiles]
                
                # Compute embeddings using persistent cached embedder
                embeddings, embed_mode = st.session_state.embedder.compute_embeddings(corpus)
                
                jd_emb = embeddings[0]
                candidate_embs = embeddings[1:]
                
                status_box.write("⚖️ Scoring candidates via gated mathematical matrix...")
                progress_bar.progress(95)
                
                # Prepare metadata for gated ranker
                candidate_metadatas = []
                for idx, profile in enumerate(valid_profiles):
                    meta = {
                        "name": profile["name"],
                        "email": profile["metadata"].get("email", "N/A"),
                        "phone": profile["metadata"].get("phone", "N/A"),
                        "skills": list(profile["metadata"].get("skills", [])),
                        "experience": profile["metadata"].get("experience", 0),
                        "matched_skills": list(profile["metadata"].get("matched_skills", [])),
                        "missing_skills": list(profile["metadata"].get("missing_skills", [])),
                        "padding_penalty_applied": profile["padding_penalty_applied"],
                        "flagged_keywords": list(profile["flagged_keywords"])
                    }
                    candidate_metadatas.append(meta)
                
                # Execute gated ranker logic
                results = rank_candidates(jd_emb, candidate_embs, candidate_metadatas, jd_metadata)
                
                # Retrieve processing logs for successful runs using dictionary lookup
                profile_by_name = {p["name"]: p for p in valid_profiles}
                for res in results:
                    p = profile_by_name[res["name"]]
                    res["parse_method"] = p["parse_method"]
                    res["nlp_mode"] = p["nlp_mode"]
                    res["processing_time_sec"] = p["elapsed_sec"]
                    res["raw_text"] = p["raw_text"]
                    res["embed_mode"] = embed_mode
                
                # Save session results
                st.session_state.results = results
                st.session_state.pipeline_run = True
                
                # Set Diagnostics
                total_pipeline_time = time.time() - pipeline_start
                st.session_state.diagnostics = {
                    "total_pipeline_time_sec": total_pipeline_time,
                    "successfully_parsed": len(valid_profiles),
                    "failed_parsed": len(failed_profiles),
                    "failed_list": [p["name"] for p in failed_profiles],
                    "embedding_mode": embed_mode
                }
                
                progress_bar.progress(100)
                status_box.empty()
                progress_bar.empty()
                st.success(f"Pipeline executed successfully in {total_pipeline_time:.2f} seconds!")
                
        except Exception as e:
            status_box.empty()
            progress_bar.empty()
            st.error(f"Execution Error: {str(e)}")
            logger.exception("Pipeline Execution crashed:")
        finally:
            # Memory Cleanup: Flush tensor cache vectors from host memory to prevent OOM
            gc.collect()
            logger.info("Memory cleanup completed: Host garbage collector run.")

# Display Results Dashboard
if st.session_state.pipeline_run and st.session_state.results:
    results = st.session_state.results
    diagnostics = st.session_state.diagnostics
    
    # 1. Diagnostic Summary Row
    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
    with diag_col1:
        st.metric("Total Ingest Time", f"{diagnostics['total_pipeline_time_sec']:.2f} s")
    with diag_col2:
        st.metric("Ingested Successfully", f"{diagnostics['successfully_parsed']} Files")
    with diag_col3:
        st.metric("Failed Ingests", f"{diagnostics['failed_parsed']} Files")
    with diag_col4:
        st.metric("Embedding Engine", diagnostics['embedding_mode'].upper())
        
    if diagnostics["failed_list"]:
        st.warning(f"⚠️ Failed to parse: {', '.join(diagnostics['failed_list'])}")
        
    st.divider()
    
    # 2. Main Dashboard Columns (Leaderboard vs scatter landscape plot)
    dash_col1, dash_col2 = st.columns([3, 2])
    
    with dash_col1:
        st.markdown('<div class="glass-card"><h4>🎯 Gated Leaderboard</h4>', unsafe_allow_html=True)
        
        # Display Explainable Dataframe
        explain_df = generate_explainable_dataframe(results)
        
        # Handle anonymized view rendering on dataframe
        if anonymize_view:
            anon_df = explain_df.copy()
            for i in range(len(anon_df)):
                anon_df.loc[i, "Candidate Name"] = f"Candidate_{anon_df.loc[i, 'Rank']:03d}"
            st.dataframe(anon_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(explain_df, use_container_width=True, hide_index=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with dash_col2:
        st.markdown('<div class="glass-card"><h4>📊 Match Score vs Experience Landscape</h4>', unsafe_allow_html=True)
        
        # Plotting DataFrame
        plot_df = pd.DataFrame({
            "Candidate Name": [f"Candidate_{rank+1:03d}" if anonymize_view else r["name"] for rank, r in enumerate(results)],
            "Experience (Yrs)": [r["experience"] for r in results],
            "Match Score (%)": [round(r["final_score"] * 100, 1) for r in results]
        })
        
        # Scatter grid plotting Match % vs Experience
        st.scatter_chart(
            plot_df,
            x="Experience (Yrs)",
            y="Match Score (%)",
            color="Candidate Name",
            size="Match Score (%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<h3>🔍 Explanatory breakdown per Profile</h3>", unsafe_allow_html=True)
    
    # Render expandable details
    for rank, res in enumerate(results):
        candidate_label = f"Candidate_{rank+1:03d}" if anonymize_view else res["name"]
        score_percentage = res["match_percentage"]
        
        expander_title = f"Rank #{rank+1} | {candidate_label} | Match Score: {score_percentage}"
        
        # Add warning visual icon if security penalty applied or gated mismatch occurred
        if res["padding_penalty_applied"]:
            expander_title += " ⚠️ [Security Warning]"
        elif res["is_gated"]:
            expander_title += " ❌ [Core Mismatch]"
            
        with st.expander(expander_title):
            col_info_left, col_info_right = st.columns([1, 2])
            
            with col_info_left:
                st.markdown("##### 📞 Contact & Details")
                if anonymize_view:
                    st.write("**Name:** `[ANONYMIZED_VIEW]`")
                    st.write("**Email:** `[ANONYMIZED_VIEW]`")
                    st.write("**Phone:** `[ANONYMIZED_VIEW]`")
                else:
                    st.write(f"**Name:** `{res['name']}`")
                    st.write(f"**Email:** `{res['email']}`")
                    st.write(f"**Phone:** `{res['phone']}`")
                    
                st.write(f"**Experience:** `{res['experience']} years`")
                
                st.divider()
                st.markdown("##### 🛡️ Security Diagnostics")
                if res["padding_penalty_applied"]:
                    st.markdown('<span class="badge-warning">Padding Penalty: Applied (-25%)</span>', unsafe_allow_html=True)
                    st.caption(f"Flagged stuffed keywords: `{', '.join(res['flagged_keywords'])}`")
                else:
                    st.success("✅ Padding Density Monitor: Safe")
                    
                if res["is_gated"]:
                    st.error("❌ Mismatched Core Technical Profile Gate applied.")
                else:
                    st.success("✅ Base Semantic Alignment Gate passed.")
                    
                st.caption(f"Parser Used: `{res['parse_method']}`")
                st.caption(f"CPU Processing Time: `{res['processing_time_sec']:.3f} seconds`")
                st.caption(f"Similarity Method: `{res['similarity_method']}`")
                
            with col_info_right:
                st.markdown("##### 🧠 Explanatory Match Feedback")
                st.info(f"💡 {res['feedback']}")
                
                st.markdown("##### 🔑 Skill Alignment Badges")
                
                # Matched Skills
                if not res["is_gated"] and res["matched_skills"]:
                    st.write("**Matched Stack:**")
                    matched_html = "".join([f'<span class="badge-matched">{s}</span>' for s in res["matched_skills"]])
                    st.markdown(matched_html, unsafe_allow_html=True)
                elif res["is_gated"]:
                    st.caption("Skills matching rewards revoked due to gated core technical profile mismatch.")
                else:
                    st.caption("No overlapping skills matched.")
                    
                # Missing Skills Gaps
                if not res["is_gated"] and res["missing_skills"]:
                    st.write("**Gaps Detected:**")
                    gap_html = "".join([f'<span class="badge-gap">{s}</span>' for s in res["missing_skills"]])
                    st.markdown(gap_html, unsafe_allow_html=True)
                elif not res["is_gated"]:
                    st.success("✅ Perfect tech-stack coverage. No skill gaps identified!")
                    
            st.divider()
            st.markdown("##### 📑 Extracted Raw Document Preview (First 2000 chars)")
            st.text_area(
                "Document Text Stream", 
                value=res["raw_text"][:2000] + ("..." if len(res["raw_text"]) > 2000 else ""),
                height=150,
                key=f"raw_text_{rank}"
            )
            
    with st.expander("🐞 System Debug logs (Raw JSON Results)"):
        debug_results = []
        for r in results:
            debug_copy = r.copy()
            if "raw_text" in debug_copy:
                del debug_copy["raw_text"]
            debug_results.append(debug_copy)
        st.json(debug_results)
