import os
import sys

# Ensure project folder is in path
sys.path.append(os.getcwd())

from src.parser import extract_text
from src.analyzer import analyze_candidate, extract_metadata

resumes_dir = "data/resumes"
jd_text = """
We are looking for a Data Scientist to design machine learning pipelines that process large-scale unstructured text data.
The ideal candidate will build semantic similarity models, implement vector search mechanisms, and deploy interactive dashboards.
You will work on cutting-edge transformer models and require 5 years of experience.
Skills: Python, PyTorch, Scikit-Learn, Docker, Git.
"""

print("=== Analyzing Local Resumes ===")
for file_name in os.listdir(resumes_dir):
    if file_name.endswith((".pdf", ".docx")):
        file_path = os.path.join(resumes_dir, file_name)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        try:
            raw_text, method = extract_text(file_bytes, file_name)
            meta = extract_metadata(raw_text)
            print(f"\nFile: {file_name}")
            print(f"Parser Method: {method}")
            print(f"Parsed Experience: {meta['experience']} years")
            print(f"Parsed Skills: {meta['skills']}")
            print(f"Raw Snippet:\n{raw_text[:200]}")
        except Exception as e:
            print(f"Failed to process {file_name}: {str(e)}")
