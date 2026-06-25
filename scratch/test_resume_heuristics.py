import os
import sys

# Ensure project folder is in path
sys.path.append(os.getcwd())

from src.analyzer import extract_name, check_is_resume, extract_metadata

print("=== Running Resume Heuristic Tests ===")

# Test 1: Code file (not a resume)
code_text = """
import os
import sys
import re
import math
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def process_files(self):
        logger.info(f"Processing files from {self.input_dir}")
        for file in os.listdir(self.input_dir):
            if file.endswith('.txt'):
                logger.info(f"Reading file: {file}")
                # perform some complex calculations
                val = math.sqrt(100)
                logger.info(f"Calculation result: {val}")

if __name__ == '__main__':
    processor = DataProcessor("/tmp/input", "/tmp/output")
    processor.process_files()
"""

email = None
phone = None
skills = []
exp = 0

is_res, reason = check_is_resume(code_text, skills, exp, email, phone)
print(f"Code Text Check: is_resume={is_res}, reason={reason}")

# Test 2: Standard Resume Text
resume_text = """
Rohan Malhotra
Software Developer
Email: rohan@example.com | Phone: +91 9999999999

Experience:
- Worked as software developer at Google for 2 years.
- Developed machine learning models using Python and PyTorch.

Education:
- B.Tech in Computer Science, IIT Delhi.

Skills: Python, PyTorch, SQL, Docker.
"""

skills = ["Python", "PyTorch", "SQL", "Docker"]
exp = 2
email = "rohan@example.com"
phone = "+91 9999999999"

is_res, reason = check_is_resume(resume_text, skills, exp, email, phone)
print(f"Resume Text Check: is_resume={is_res}, reason={reason}")
print(f"Extracted Name: {extract_name(resume_text)}")

# Test 3: Short non-resume text
short_text = "This is a random document for testing purpose. Not a resume."
is_res, reason = check_is_resume(short_text, [], 0, None, None)
print(f"Short Text Check: is_resume={is_res}, reason={reason}")

# Test 4: Name with "Name:" prefix
resume_text_with_prefix = """
Name: Rohan Malhotra
Software Developer
Email: rohan@example.com

Summary:
Experience in Python coding.
"""
print(f"Extracted Name with prefix: {extract_name(resume_text_with_prefix)}")
