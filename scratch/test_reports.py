import os
import sys

# Ensure project folder is in path
sys.path.append(os.getcwd())

from app import generate_excel_report, generate_pdf_report

print("=== Running Reports Generation Test ===")

mock_results = [
    {
        "name": "Rohan Malhotra",
        "final_score": 0.445,
        "base_score": 0.445,
        "match_percentage": "44.5%",
        "experience": 3,
        "email": "rohan@example.com",
        "phone": "+91 9999999999",
        "is_gated": True,
        "is_resume": True,
        "padding_penalty_applied": False,
        "matched_skills": ["Python", "PyTorch", "SQL"],
        "missing_skills": ["Docker", "Git", "Kubernetes"],
        "feedback": "Mismatched core technical profile."
    },
    {
        "name": "Preeti Sharma",
        "final_score": 0.82,
        "base_score": 0.72,
        "match_percentage": "82.0%",
        "experience": 8,
        "email": "preeti@example.com",
        "phone": "N/A",
        "is_gated": False,
        "is_resume": True,
        "padding_penalty_applied": False,
        "matched_skills": ["Java", "Spring", "Oracle"],
        "missing_skills": [],
        "feedback": "Excellent candidate."
    }
]

try:
    print("Generating Excel report...")
    excel_bytes = generate_excel_report(mock_results)
    print(f"Excel report generated. Size: {len(excel_bytes)} bytes")
    
    print("Generating PDF report...")
    pdf_bytes = generate_pdf_report(mock_results, "Mock JD text requirements")
    print(f"PDF report generated. Size: {len(pdf_bytes)} bytes")
    
    print("=== Reports Test PASSED ===")
except Exception as e:
    print(f"Failed: {str(e)}")
    sys.exit(1)
