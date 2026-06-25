import streamlit as st
import pandas as pd
import os
import time
import logging
import concurrent.futures
import gc  # Garbage Collection
import io
from fpdf import FPDF

# Import pipeline components
from src.parser import extract_text
from src.analyzer import clean_text, extract_metadata, analyze_candidate
from src.embedder import DocumentEmbedder
from src.ranker import rank_candidates, generate_explainable_dataframe

# Helper classes and functions for exporting reports
def clean_pdf_text(text):
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",  # curly single quotes
        "\u201c": '"', "\u201d": '"',  # curly double quotes
        "\u2014": "-", "\u2013": "-",  # em/en dashes
        "\u2022": "-", "\u00b7": "-",  # bullets
        "\u2192": "->",                # arrows
        "\u00ae": "(R)", "\u00a9": "(C)",
    }
    cleaned = text
    for uni_char, ascii_char in replacements.items():
        cleaned = cleaned.replace(uni_char, ascii_char)
    # Encode to latin-1 and ignore unknown characters to prevent FPDF crashes
    return cleaned.encode('latin-1', errors='replace').decode('latin-1')

class TalentMatchPDF(FPDF):
    def header(self):
        # Top banner decoration
        self.set_fill_color(79, 70, 229) # Premium Indigo color
        self.rect(0, 0, 210, 10, 'F')
        
        self.ln(5)
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(31, 41, 55) # Dark gray text
        self.cell(0, 10, 'TalentMatch AI Evaluation Report', 0, 1, 'L')
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(107, 114, 128)
        self.cell(0, 5, 'Enterprise Sourcing & Discovery Hub | Gated Semantic Evaluation', 0, 1, 'L')
        self.ln(5)
        self.set_draw_color(229, 231, 235) # Light gray divider
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(results, jd_text):
    pdf = TalentMatchPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # JD info summary
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 6, 'Target Job Requirements Summary:', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(55, 65, 81)
    jd_summary = (jd_text[:300] + "...") if len(jd_text) > 300 else jd_text
    pdf.multi_cell(0, 5, clean_pdf_text(jd_summary.strip()))
    pdf.ln(5)
    
    # Leaderboard title
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 6, 'Gated Candidate Leaderboard:', 0, 1, 'L')
    pdf.ln(2)
    
    # Table Header
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(243, 244, 246) # Light gray fill
    pdf.set_text_color(55, 65, 81)
    
    # Col widths
    widths = [10, 45, 20, 20, 15, 35, 45]
    headers = ["Rank", "Candidate Name", "Match %", "Base Score", "Exp (Yrs)", "Security Status", "Contact (Email)"]
    
    for w, h in zip(widths, headers):
        pdf.cell(w, 8, h, 1, 0, 'C', True)
    pdf.ln(8)
    
    # Table Rows
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(75, 85, 99)
    
    for rank, res in enumerate(results):
        status = "Passed Gate"
        if not res.get("is_resume", True):
            status = "Invalid"
        elif res["is_gated"]:
            status = "Gated"
        elif res["padding_penalty_applied"]:
            status = "Security Flag"
            
        email_str = str(res.get("email")) if res.get("email") else "N/A"
            
        pdf.cell(widths[0], 7, str(rank + 1), 1, 0, 'C')
        pdf.cell(widths[1], 7, clean_pdf_text(res["name"][:24]), 1, 0, 'L')
        pdf.cell(widths[2], 7, clean_pdf_text(res["match_percentage"]), 1, 0, 'C')
        pdf.cell(widths[3], 7, f"{res['base_score']:.3f}", 1, 0, 'C')
        pdf.cell(widths[4], 7, f"{res['experience']} yrs", 1, 0, 'C')
        
        # Color code security status cell
        if status == "Invalid":
            pdf.set_text_color(220, 38, 38) # Red
        elif status == "Gated":
            pdf.set_text_color(245, 158, 11) # Orange
        else:
            pdf.set_text_color(75, 85, 99)
            
        pdf.cell(widths[5], 7, status, 1, 0, 'C')
        pdf.set_text_color(75, 85, 99)
        pdf.cell(widths[6], 7, clean_pdf_text(email_str[:25]), 1, 1, 'L')
        
    pdf.ln(8)
    
    # Detailed candidate pages
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 6, 'Detailed Candidate Evaluations:', 0, 1, 'L')
    pdf.ln(3)
    
    for rank, res in enumerate(results):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(0, 6, clean_pdf_text(f'#{rank + 1} - {res["name"]} (Match Score: {res["match_percentage"]})'), 0, 1, 'L')
        
        email_str = str(res.get("email")) if res.get("email") else "N/A"
        phone_str = str(res.get("phone")) if res.get("phone") else "N/A"
        
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(50, 5, clean_pdf_text(f'Email: {email_str}'), 0, 0, 'L')
        pdf.cell(50, 5, clean_pdf_text(f'Phone: {phone_str}'), 0, 0, 'L')
        pdf.cell(50, 5, clean_pdf_text(f'Experience: {res["experience"]} years'), 0, 1, 'L')
        
        # Skills
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 4, 'Matched Stack:', 0, 1, 'L')
        pdf.set_font('Helvetica', '', 8)
        matched_str = ", ".join(res["matched_skills"]) if res["matched_skills"] else "None"
        pdf.multi_cell(190, 4, clean_pdf_text(matched_str))
        pdf.ln(1)
        
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 4, 'Missing Stack:', 0, 1, 'L')
        pdf.set_font('Helvetica', '', 8)
        missing_str = ", ".join(res["missing_skills"]) if res["missing_skills"] else "Perfect Tech-Stack Match!"
        pdf.multi_cell(190, 4, clean_pdf_text(missing_str))
        pdf.ln(1)
        
        # Projects evaluation in PDF
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 4, 'Projects & Complexity:', 0, 1, 'L')
        pdf.set_font('Helvetica', '', 8)
        proj_details = res.get("project_details", {})
        proj_score_pct = f"{res.get('project_score', 0.0) * 100:.1f}%"
        proj_count = proj_details.get("project_count", 0)
        proj_skills_str = ", ".join(proj_details.get("integrated_skills", [])) if proj_details.get("integrated_skills") else "None"
        complexity_val = proj_details.get("complexity_score", 0.0)
        complexity_str = "High" if complexity_val >= 0.7 else "Medium" if complexity_val >= 0.4 else "Low"
        proj_summary_str = f"Project Score: {proj_score_pct} | Projects Detected: {proj_count} | Complexity: {complexity_str} | Integrated Stack: {proj_skills_str}"
        pdf.multi_cell(190, 4, clean_pdf_text(proj_summary_str))
        pdf.ln(1)
        
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 4, 'Evaluation Feedback:', 0, 1, 'L')
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(107, 114, 128)
        pdf.multi_cell(190, 4, clean_pdf_text(res["feedback"]))
        pdf.set_text_color(55, 65, 81)
        
        pdf.ln(4)
        pdf.set_draw_color(243, 244, 246)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
    return bytes(pdf.output())
 
def generate_excel_report(results):
    buffer = io.BytesIO()
    excel_data = []
    for rank, res in enumerate(results):
        status = "Passed Gate"
        if not res.get("is_resume", True):
            status = "Invalid Document"
        elif res["is_gated"]:
            status = "Gated (Low Alignment)"
        elif res["padding_penalty_applied"]:
            status = "Security Flag (Padding)"
            
        proj_details = res.get("project_details", {})
        proj_score_pct = f"{res.get('project_score', 0.0) * 100:.1f}%"
        proj_count = proj_details.get("project_count", 0)
        proj_skills_str = ", ".join(proj_details.get("integrated_skills", [])) if proj_details.get("integrated_skills") else "None"
        complexity_val = proj_details.get("complexity_score", 0.0)
        complexity_str = "High" if complexity_val >= 0.7 else "Medium" if complexity_val >= 0.4 else "Low"

        excel_data.append({
            "Rank": rank + 1,
            "Candidate Name": res["name"],
            "Match Score (%)": res["match_percentage"],
            "Base Semantic Score": round(res["base_score"], 3),
            "Experience (Yrs)": res["experience"],
            "Email": res.get("email") if res.get("email") else "N/A",
            "Phone": res.get("phone") if res.get("phone") else "N/A",
            "Security Status": status,
            "Matched Skills": ", ".join(res["matched_skills"]),
            "Missing Skills": ", ".join(res["missing_skills"]),
            "Project Score (%)": proj_score_pct,
            "Projects Count": proj_count,
            "Project Complexity": complexity_str,
            "Project Integrated Skills": proj_skills_str,
            "Feedback": res["feedback"]
        })
    df = pd.DataFrame(excel_data)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Leaderboard")
    return buffer.getvalue()

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
    
    anonymize_view = st.toggle("Anonymized Sourcing View", value=False,
                               help="Hides candidate names and contact data from expanded profile views to ensure absolute technical objectivity.")
    
    st.markdown('<div class="sidebar-header">⚙️ System Configurations</div>', unsafe_allow_html=True)
    
    use_transformer = st.toggle("Enable Local Transformer Model", value=True,
                                help="Toggle local SentenceTransformer ('all-MiniLM-L6-v2'). If disabled, runs fallback TF-IDF vectorizer.")
    
    gate_threshold = st.slider("Semantic Alignment Gate Threshold", min_value=0.10, max_value=0.60, value=0.30, step=0.05,
                               help="The minimum semantic similarity required for the candidate to pass the core qualifications gate. Profiles below this are marked as 'Gated (Low Alignment)'.")
    
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
                        "name": profile["metadata"].get("candidate_name", profile["name"]),
                        "email": profile["metadata"].get("email", "N/A"),
                        "phone": profile["metadata"].get("phone", "N/A"),
                        "skills": list(profile["metadata"].get("skills", [])),
                        "experience": profile["metadata"].get("experience", 0),
                        "matched_skills": list(profile["metadata"].get("matched_skills", [])),
                        "missing_skills": list(profile["metadata"].get("missing_skills", [])),
                        "padding_penalty_applied": profile["padding_penalty_applied"],
                        "flagged_keywords": list(profile["flagged_keywords"]),
                        "is_resume": profile["metadata"].get("is_resume", True),
                        "resume_validation_reason": profile["metadata"].get("resume_validation_reason", ""),
                        "project_score": profile["metadata"].get("project_score", 0.0),
                        "project_details": profile["metadata"].get("project_details", {})
                    }
                    candidate_metadatas.append(meta)
                
                # Save session data for dynamic re-ranking
                st.session_state.jd_emb = jd_emb
                st.session_state.candidate_embs = candidate_embs
                st.session_state.candidate_metadatas = candidate_metadatas
                st.session_state.jd_metadata = jd_metadata
                st.session_state.valid_profiles = valid_profiles
                st.session_state.embed_mode = embed_mode
                st.session_state.pipeline_run = True
                
                # Set Diagnostics
                total_pipeline_time = time.time() - pipeline_start
                st.session_state.diagnostics = {
                    "total_pipeline_time_sec": total_pipeline_time,
                    "successfully_parsed": len(valid_profiles),
                    "failed_parsed": len(failed_profiles),
                    "failed_list": [p["name"] for p in failed_profiles],
                    "embedding_mode": embed_mode,
                    "processed_filenames": [f.name for f in uploaded_resumes] if uploaded_resumes else []
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
if st.session_state.pipeline_run:
    # Check if the list of currently uploaded resumes has changed from the processed set
    current_filenames = set(f.name for f in uploaded_resumes) if uploaded_resumes else set()
    processed_filenames = set(st.session_state.diagnostics.get('processed_filenames', []))
    if current_filenames != processed_filenames:
        st.warning("⚠️ **File Queue Changed:** The list of uploaded resumes has changed. Please click the **👑 Execute Sourcing & Hybrid Matching Engine** button above to update the results for all candidates.")
    # Dynamic re-ranking based on current gate_threshold slider
    results = rank_candidates(
        st.session_state.jd_emb,
        st.session_state.candidate_embs,
        st.session_state.candidate_metadatas,
        st.session_state.jd_metadata,
        gate_threshold=gate_threshold
    )
    # Restore runtime keys
    for res in results:
        p = st.session_state.valid_profiles[res["candidate_index"]]
        res["parse_method"] = p["parse_method"]
        res["nlp_mode"] = p["nlp_mode"]
        res["processing_time_sec"] = p["elapsed_sec"]
        res["raw_text"] = p["raw_text"]
        res["embed_mode"] = st.session_state.embed_mode
        
    st.session_state.results = results
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

    # 1.5. Spotlight Podium Cards (Top 3 Matches)
    st.markdown("### 🏆 Top Talent Matches")
    top_candidates = results[:3]
    
    cols = st.columns(len(top_candidates))
    for idx, cand in enumerate(top_candidates):
        with cols[idx]:
            if idx == 0:
                border_color = "#fbbf24" # Gold
                badge_icon = "🥇"
                rank_name = "Rank 1 (Top Match)"
            elif idx == 1:
                border_color = "#9ca3af" # Silver
                badge_icon = "🥈"
                rank_name = "Rank 2"
            else:
                border_color = "#cd7f32" # Bronze
                badge_icon = "🥉"
                rank_name = "Rank 3"
                
            rank = idx + 1
            name_label = f"Candidate_{rank:03d}" if anonymize_view else cand["name"]
            
            status_text = "Passed Gate"
            if not cand["is_resume"]:
                status_text = "Invalid Document"
            elif cand["padding_penalty_applied"]:
                status_text = "Security Flag (Padding)"
            elif cand["is_gated"]:
                status_text = "Gated (Low Alignment)"
                
            skills_matched = len(cand['matched_skills'])
            skills_total = len(cand['matched_skills']) + len(cand['missing_skills'])
            
            st.markdown(f"""
            <div style="
                border: 2px solid {border_color};
                border-radius: 12px;
                padding: 15px;
                background: rgba(17, 24, 39, 0.6);
                backdrop-filter: blur(8px);
                margin-bottom: 15px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            ">
                <div style="font-size: 1.25rem; font-weight: 700; color: #f3f4f6; margin-bottom: 5px;">
                    {badge_icon} {name_label}
                </div>
                <div style="color: {border_color}; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 10px;">
                    {rank_name}
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Match Score:</span>
                    <span style="color: #f3f4f6; font-weight: 700; font-size: 1.1rem;">{cand['match_percentage']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Experience:</span>
                    <span style="color: #f3f4f6; font-weight: 600;">{cand['experience']} Years</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Skills Match:</span>
                    <span style="color: #f3f4f6; font-weight: 600;">{skills_matched} / {skills_total}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Project Score:</span>
                    <span style="color: #f3f4f6; font-weight: 600;">{cand['project_score'] * 100:.1f}% ({cand['project_details'].get('project_count', 0)} Projs)</span>
                </div>
                <div style="
                    text-align: center;
                    background: {border_color}1a;
                    color: {border_color};
                    padding: 4px;
                    border-radius: 6px;
                    font-size: 0.8rem;
                    font-weight: 700;
                    border: 1px solid {border_color}33;
                ">
                    {status_text.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
            
        # Add Excel and PDF download option section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h5>📥 Export discovery Results</h5>", unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            try:
                excel_bytes = generate_excel_report(results)
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_bytes,
                    file_name="TalentMatch_Leaderboard_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate Excel: {str(e)}")
        with export_col2:
            try:
                pdf_bytes = generate_pdf_report(results, jd_text)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="TalentMatch_Evaluation_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {str(e)}")
            
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
        if not res.get("is_resume", True):
            expander_title += " ❌ [Not a Resume]"
        elif res["padding_penalty_applied"]:
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
                    
                if not res.get("is_resume", True):
                    st.error(f"❌ Resume Validity Check: FAILED ({res.get('resume_validation_reason')})")
                elif res["is_gated"]:
                    st.error("❌ Mismatched Core Technical Profile Gate applied.")
                else:
                    st.success("✅ Base Semantic Alignment Gate passed.")
                    
                st.caption(f"Parser Used: `{res['parse_method']}`")
                st.caption(f"CPU Processing Time: `{res['processing_time_sec']:.3f} seconds`")
                st.caption(f"Similarity Method: `{res['similarity_method']}`")
                
            with col_info_right:
                st.markdown("##### 🧠 Explanatory Match Feedback")
                st.info(f"💡 {res['feedback']}")
                
                if res["is_gated"] and res.get("is_resume", True):
                    st.warning(f"⚠️ Note: Skill and experience weights were not added to final score due to gated semantic alignment mismatch (<{gate_threshold:.2f} similarity).")
                
                st.markdown("##### 🔑 Skill Alignment Badges")
                
                # Matched Skills
                if res["matched_skills"]:
                    st.write("**Matched Stack:**")
                    matched_html = "".join([f'<span class="badge-matched">{s}</span>' for s in res["matched_skills"]])
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.caption("No overlapping skills matched.")
                    
                # Missing Skills Gaps
                if res["missing_skills"]:
                    st.write("**Gaps Detected:**")
                    gap_html = "".join([f'<span class="badge-gap">{s}</span>' for s in res["missing_skills"]])
                    st.markdown(gap_html, unsafe_allow_html=True)
                else:
                    st.success("✅ Perfect tech-stack coverage. No skill gaps identified!")
                    
                # Projects Analysis Breakdown
                st.markdown("##### 📁 Project Integration & Complexity")
                proj_details = res.get("project_details", {})
                proj_score = res.get("project_score", 0.0)
                proj_count = proj_details.get("project_count", 0)
                proj_skills = proj_details.get("integrated_skills", [])
                complexity_val = proj_details.get("complexity_score", 0.0)
                
                # Complexity rating
                if complexity_val >= 0.7:
                    complexity_badge = '<span style="color:#10b981; font-weight:700;">💎 High Complexity (Architecture, Cloud, APIs, Optimization)</span>'
                elif complexity_val >= 0.4:
                    complexity_badge = '<span style="color:#f59e0b; font-weight:700;">⚙️ Medium Complexity (Database, APIs, or Models)</span>'
                else:
                    complexity_badge = '<span style="color:#9ca3af; font-weight:600;">📁 Low Complexity (Basic scripts / static projects)</span>'
                
                st.write(f"**Project Score:** `{proj_score * 100:.1f}%`")
                st.write(f"**Project Count:** `{proj_count}` projects detected")
                st.write(f"**Complexity Rating:** {complexity_badge}", unsafe_allow_html=True)
                
                if proj_skills:
                    st.write("**Skills Integrated in Projects:**")
                    proj_skills_html = "".join([f'<span class="badge-matched" style="background: rgba(99, 102, 241, 0.12); color: #818cf8; border-color: rgba(99, 102, 241, 0.3);">{s}</span>' for s in proj_skills])
                    st.markdown(proj_skills_html, unsafe_allow_html=True)
                else:
                    st.caption("No target skills detected in the project descriptions.")
            st.divider()
            st.markdown("##### 📑 Extracted Document Text Preview (Scroll to view full resume)")
            st.text_area(
                "Extracted text used by the semantic matching engine", 
                value=res["raw_text"],
                height=250,
                key=f"raw_text_{rank}"
            )
