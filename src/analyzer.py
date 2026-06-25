import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import security engines
from src.security import anonymize_text, detect_and_clean_padding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to download NLTK data programmatically
NLTK_RESOURCES = [
    'punkt', 
    'punkt_tab', 
    'stopwords', 
    'wordnet', 
    'averaged_perceptron_tagger', 
    'averaged_perceptron_tagger_eng'
]
NLTK_AVAILABLE = False

try:
    for resource in NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK resource '{resource}': {str(e)}")
            
    # Verify resources can be loaded
    stopwords.words('english')
    WordNetLemmatizer()
    word_tokenize("test text")
    nltk.pos_tag(["test"])
    NLTK_AVAILABLE = True
    logger.info("NLTK successfully loaded and verified with POS tagger.")
except Exception as e:
    logger.warning(f"NLTK setup failed: {str(e)}. Falling back to regex-based processing.")
    NLTK_AVAILABLE = False

# Common skills vocabulary
COMMON_SKILLS = [
    "python", "java", "c\\+\\+", "c#", "javascript", "typescript", "ruby", "php", "go", "rust", "scala", "kotlin", "swift", "r", "sql", "nosql", "bash", "shell",
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "cv", "reinforcement learning", 
    "data science", "data analysis", "neural networks", "llm", "large language models", "generative ai", "transformers",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "nltk", "spacy", "gensim",
    "flask", "django", "fastapi", "spring", "spring boot", "react", "angular", "vue", "node\\.js", "express", "next\\.js", "jquery",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "sqlite", "oracle", "dynamodb",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "git", "github", "gitlab", "jenkins", "ci/cd", "terraform", "ansible",
    "agile", "scrum", "rest api", "graphql", "microservices", "system design", "spark", "hadoop", "hive", "kafka"
]

SKILL_DISPLAY_MAP = {
    "python": "Python",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "ci/cd": "CI/CD",
    "rest api": "REST API",
    "mongodb": "MongoDB",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "fastapi": "FastAPI",
    "node.js": "Node.js",
    "node\\.js": "Node.js",
    "github": "GitHub",
    "gitlab": "GitLab",
    "gcp": "GCP",
    "aws": "AWS",
    "nlp": "NLP",
    "html": "HTML",
    "css": "CSS",
    "sql": "SQL",
    "nosql": "NoSQL",
    "java": "Java",
    "git": "Git",
    "spring boot": "Spring Boot",
    "spring": "Spring"
}

def clean_text_primary(text):
    """NLTK-based cleaning, tokenization, stopword removal, and lemmatization."""
    if not NLTK_AVAILABLE:
        raise RuntimeError("NLTK is not available for text cleaning.")
    
    cleaned = text.lower()
    cleaned = re.sub(r'[\u2022\u00b7\u25cf\u25fe\u25aa\u25ab\u2b24\u25c6\u27a4\-•\*+]', ' ', cleaned)
    cleaned = re.sub(r'[^a-z0-9\s#+.]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return " ".join(lemmatized_tokens)

def clean_text_fallback(text):
    """Fallback clean text pipeline using only basic python string methods and raw regex."""
    cleaned = text.lower()
    cleaned = re.sub(r'[\u2022\u00b7\u25cf\u25fe\u25aa\u25ab\u2b24\u25c6\u27a4\-•\*+]', ' ', cleaned)
    cleaned = re.sub(r'[^a-z0-9\s#+.]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    tokens = cleaned.split()
    basic_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
        'should', 'now'
    }
    
    filtered_tokens = [w for w in tokens if w not in basic_stopwords]
    stemmed_tokens = []
    for w in filtered_tokens:
        if w.endswith('ies') and len(w) > 3:
            stemmed_tokens.append(w[:-3] + 'y')
        elif w.endswith('es') and len(w) > 3 and w[-3] in 'shxoz':
            stemmed_tokens.append(w[:-2])
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 2:
            stemmed_tokens.append(w[:-1])
        else:
            stemmed_tokens.append(w)
            
    return " ".join(stemmed_tokens)

def clean_text(text):
    if NLTK_AVAILABLE:
        try:
            return clean_text_primary(text), "nltk"
        except Exception as e:
            logger.error(f"NLTK cleaning failed: {str(e)}. Falling back to regex cleaning.")
            
    return clean_text_fallback(text), "regex-fallback"

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text):
    phone_pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,13})'
    match = re.search(phone_pattern, text)
    return match.group(0).strip() if match else None

def extract_skills(text):
    found_skills = []
    lower_text = text.lower()
    for skill in COMMON_SKILLS:
        if "c++" in skill:
            pattern = r'\bc\+\+(?:\b|(?=\s))'
        elif "c#" in skill:
            pattern = r'\bc#(?:\b|(?=\s))'
        elif "node.js" in skill or "node\\.js" in skill:
            pattern = r'\bnode\.js\b'
        elif "ci/cd" in skill:
            pattern = r'\bci/cd\b'
        elif "rest api" in skill:
            pattern = r'\brest api\b'
        else:
            pattern = r'\b' + skill + r'\b'
            
        if re.search(pattern, lower_text):
            clean_skill = skill.replace('\\', '')
            display_name = SKILL_DISPLAY_MAP.get(clean_skill, clean_skill.upper() if len(clean_skill) <= 4 else clean_skill.title())
            found_skills.append(display_name)
            
    return sorted(list(set(found_skills)))

def extract_experience(text):
    patterns = [
        r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s*[:\-]?\s*(\d+)\s*\+?\s*years?',
        r'(\d+)\s*\+?\s*years?\s+(?:in|working|professional|industry)',
        r'(\d+)\s*\+?\s*yrs?\s+(?:of\s+)?experience',
        r'(\d+)\s*\+?\s*years?\s+exp\b'
    ]
    
    found_years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                found_years.append(int(m))
            except ValueError:
                continue
                
    if found_years:
        return max(found_years)
        
    sec_pattern = r'(?:experience|exp)\s*[:\-]\s*(\d{1,2})\b'
    matches = re.findall(sec_pattern, text, re.IGNORECASE)
    for m in matches:
        try:
            return int(m)
        except ValueError:
            continue
            
    return 0

def extract_name(text):
    """
    Extracts candidate name from the first non-empty line of text.
    Filters out common section headers, cleans prefixes, and ensures it is not an invalid keyword.
    """
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped:
            lower_line = stripped.lower()
            is_heading = any(h in lower_line for h in ["summary", "profile", "experience", "education", "skills", "objective", "contact", "email", "phone"])
            if not is_heading and len(stripped) < 60 and len(stripped.split()) >= 2:
                # Clean prefix patterns like "Name:", "Candidate Name:"
                prefix_pattern = r'^(?:name|candidate|candidate name|cv|resume|curriculum vitae)\s*[:\-]\s*'
                name_candidate = re.sub(prefix_pattern, '', stripped, flags=re.IGNORECASE).strip()
                
                # Remove leading/trailing symbols
                name_candidate = re.sub(r'^[\s\-:*•●]+|[\s\-:*•●]+$', '', name_candidate).strip()
                
                # Reject invalid names
                invalid_names = {"curriculum vitae", "resume", "cv", "portfolio", "biodata", "cover letter"}
                if name_candidate.lower() in invalid_names or len(name_candidate.split()) < 2:
                    continue
                
                return name_candidate
    return "Unknown Candidate"

def check_is_resume(text, skills, experience, email, phone):
    """
    Heuristic check to determine if the document behaves like a valid resume.
    """
    text_lower = text.lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 1. Word count threshold (Resumes are rarely under 40 words)
    words = re.findall(r'\b\w+\b', text_lower)
    if len(words) < 40:
        return False, f"Document is too short to be a valid resume ({len(words)} words found, minimum is 40)."
        
    # 2. Check for code syntax to filter out programming scripts
    code_indicators = [
        r'^\s*import\s+\w+',
        r'^\s*from\s+\w+\s+import',
        r'^\s*def\s+\w+\s*\(',
        r'^\s*class\s+\w+\s*[\(:]',
        r'#\s*type:\s*\w+',
        r'if\s+__name__\s*==\s*[\'"]__main__[\'"]'
    ]
    code_line_matches = 0
    for line in lines:
        if any(re.match(pat, line) for pat in code_indicators):
            code_line_matches += 1
            
    if code_line_matches >= 3 or (len(lines) > 0 and code_line_matches / len(lines) > 0.15):
        return False, "Document matches source code structure (contains python import/class/function definitions)."
        
    # 3. Check for JSON / XML / HTML structures
    if text_lower.startswith("{") and text_lower.endswith("}"):
        return False, "Document appears to be a raw JSON structure, not a resume."
    if text_lower.startswith("<html") or text_lower.startswith("<!doctype html"):
        return False, "Document appears to be HTML source code, not a resume."
        
    # 4. Check for key resume sections
    section_patterns = {
        "education": [r'\beducation\b', r'\bacademic\b', r'\buniversity\b', r'\bcollege\b', r'\bdegree\b'],
        "experience": [r'\bexperience\b', r'\bwork\b', r'\bemployment\b', r'\bhistory\b', r'\bprofessional\b'],
        "skills": [r'\bskills\b', r'\btechnologies\b', r'\btools\b', r'\bexpertise\b'],
        "projects": [r'\bprojects\b', r'\bpublications\b', r'\baccomplishments\b']
    }
    
    found_sections = set()
    for sec, patterns in section_patterns.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                found_sections.add(sec)
                break
                
    # If the document has none of the standard resume section terms
    if len(found_sections) < 2:
        resume_indicators = [
            "experience", "education", "skills", "employment", "work", "project", 
            "summary", "history", "academic", "profile", "career", "contact", 
            "qualification", "certification", "achievement", "objective", "publication",
            "internship", "volunteer", "job", "developer", "engineer", "designer", "manager"
        ]
        matched_indicators = [kw for kw in resume_indicators if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        if len(matched_indicators) < 3:
            return False, "Document lacks standard resume section markers and vocabulary."
            
    # 5. Check for contact info and profile details
    if not email and not phone and not skills and experience == 0:
        return False, "Lacks contact details (email/phone), skill tags, experience details, and standard resume headers."
            
    return True, ""

def extract_metadata(text):
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text),
        "candidate_name": extract_name(text)
    }

def analyze_candidate(raw_text, jd_text):
    """
    Main entry point for candidate analysis.
    1. Scan for keyword stuffing/padding and strip duplicate words.
    2. Anonymize/De-bias the text.
    3. Tokenize and extract skills/metadata.
    """
    # 1. Anti-Cheat: Padding detection and keyword cleaning
    sanitized_text, padding_flag, flagged_keywords = detect_and_clean_padding(raw_text)
    
    # 2. AI De-biasing & Anonymization
    anonymized_text = anonymize_text(sanitized_text, nltk_available=NLTK_AVAILABLE)
    
    # 3. Clean text for semantic representation
    cleaned_text, nlp_mode = clean_text(anonymized_text)
    
    # 4. Extract metadata (email, phone, skills, experience)
    # Contact fields are extracted from raw_text before anonymization, skills are read from clean context
    meta = extract_metadata(raw_text)
    jd_meta = extract_metadata(jd_text)
    
    # 5. Skill Gap Analysis
    cand_skills = set(meta["skills"])
    jd_skills = set(jd_meta["skills"])
    
    matched_skills = sorted(list(jd_skills.intersection(cand_skills)))
    missing_skills = sorted(list(jd_skills.difference(cand_skills)))
    
    meta["matched_skills"] = matched_skills
    meta["missing_skills"] = missing_skills
    meta["total_jd_skills_count"] = len(jd_skills)
    meta["skill_match_ratio"] = len(matched_skills) / len(jd_skills) if jd_skills else 1.0
    
    # 6. Resume Validation Check
    is_resume, resume_validation_reason = check_is_resume(
        raw_text, meta["skills"], meta["experience"], meta["email"], meta["phone"]
    )
    meta["is_resume"] = is_resume
    meta["resume_validation_reason"] = resume_validation_reason
    
    return cleaned_text, meta, nlp_mode, padding_flag, flagged_keywords
