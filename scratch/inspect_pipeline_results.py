import os
import sys

# Ensure project folder is in path
sys.path.append(os.getcwd())

from src.parser import extract_text
from src.analyzer import analyze_candidate, extract_metadata
from src.embedder import DocumentEmbedder
from src.ranker import rank_candidates, generate_explainable_dataframe

resumes_dir = "data/resumes"
jd_text = """
We are looking for a Data Scientist to design machine learning pipelines that process large-scale unstructured text data.
The ideal candidate will build semantic similarity models, implement vector search mechanisms, and deploy interactive dashboards.
You will work on cutting-edge transformer models and require 5 years of experience.
Skills: Python, PyTorch, Scikit-Learn, Docker, Git.
"""

# Ingest JD
clean_jd_text = jd_text.strip()
jd_metadata = extract_metadata(jd_text)

processed_profiles = []
for file_name in os.listdir(resumes_dir):
    if file_name.endswith((".pdf", ".docx")):
        file_path = os.path.join(resumes_dir, file_name)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        try:
            raw_text, parse_method = extract_text(file_bytes, file_name)
            cleaned_text, metadata, nlp_mode, padding_flag, flagged_keywords = analyze_candidate(raw_text, jd_text)
            processed_profiles.append({
                "success": True,
                "name": os.path.splitext(file_name)[0],
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "metadata": metadata,
                "parse_method": parse_method,
                "nlp_mode": nlp_mode,
                "padding_penalty_applied": padding_flag,
                "flagged_keywords": flagged_keywords,
                "elapsed_sec": 0.1
            })
        except Exception as e:
            print(f"Failed {file_name}: {e}")

valid_profiles = [p for p in processed_profiles if p["success"]]

print("\n--- Valid Profiles Order ---")
for idx, p in enumerate(valid_profiles):
    print(f"Index {idx}: Name={p['name']}, Experience={p['metadata']['experience']}, Skills={p['metadata']['skills']}")

# Compute embeddings
embedder = DocumentEmbedder(use_local_transformer=True)
corpus = [clean_jd_text] + [p["cleaned_text"] for p in valid_profiles]
embeddings, embed_mode = embedder.compute_embeddings(corpus)

jd_emb = embeddings[0]
candidate_embs = embeddings[1:]

# Prepare metadata
candidate_metadatas = []
for idx, profile in enumerate(valid_profiles):
    meta = profile["metadata"]
    meta["name"] = profile["name"]
    meta["email"] = profile["metadata"].get("email", "N/A")
    meta["phone"] = profile["metadata"].get("phone", "N/A")
    meta["padding_penalty_applied"] = profile["padding_penalty_applied"]
    meta["flagged_keywords"] = profile["flagged_keywords"]
    candidate_metadatas.append(meta)

# Run ranker
results = rank_candidates(jd_emb, candidate_embs, candidate_metadatas, jd_metadata)

print("\n--- Ranking Results ---")
for idx, res in enumerate(results):
    print(f"Rank {idx+1}:")
    print(f"  Name: {res['name']}")
    print(f"  Final Score: {res['final_score']:.3f} ({res['match_percentage']})")
    print(f"  Base Score: {res['base_score']:.3f}")
    print(f"  Experience: {res['experience']} years")
    print(f"  Feedback: {res['feedback']}")
