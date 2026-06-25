import re
import logging

logger = logging.getLogger(__name__)

# List of common skills to match from analyzer to avoid name/skill collisions
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

def detect_and_clean_padding(text):
    """
    Scans the incoming text stream for keyword stuffing / resume padding.
    If a technical keyword's frequency exceeds 8% of the document's word count:
    1. Flag candidate for a Suspected Resume Padding Violation.
    2. Strip duplicate occurrences of the keyword, keeping only the first.
    3. Apply score penalty indicator.
    """
    # Count total words in text
    words = re.findall(r'\b[a-zA-Z0-9#+.]+\b', text.lower())
    total_words = len(words)
    
    if total_words == 0:
        return text, False, []
        
    flagged_keywords = []
    penalty_applied = False
    
    # Check frequency of each skill keyword
    for skill in COMMON_SKILLS:
        clean_skill = skill.replace('\\', '')
        # Formulate boundary regex
        if "c++" in clean_skill:
            pattern = r'\bc\+\+(?:\b|(?=\s))'
        elif "c#" in clean_skill:
            pattern = r'\bc#(?:\b|(?=\s))'
        else:
            pattern = r'\b' + re.escape(clean_skill) + r'\b'
            
        matches = re.findall(pattern, text, re.IGNORECASE)
        count = len(matches)
        
        # If density exceeds 8%
        if count / total_words > 0.08:
            flagged_keywords.append(clean_skill)
            penalty_applied = True
            logger.warning(f"SECURITY ALERT: Keyword stuffing detected for '{clean_skill}' (Density: {count}/{total_words} = {count/total_words:.2%})")
            
    cleaned_text = text
    if penalty_applied:
        # Strip duplicate occurrences of flagged keywords, leaving only the first occurrence
        for kw in flagged_keywords:
            if "c++" in kw:
                pattern = r'\bc\+\+(?:\b|(?=\s))'
            elif "c#" in kw:
                pattern = r'\bc#(?:\b|(?=\s))'
            else:
                pattern = r'\b' + re.escape(kw) + r'\b'
                
            # Iterate and replace all but the first occurrence
            matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
            if len(matches) > 1:
                # Keep the first match, replace the rest with empty space
                # Do this backwards to preserve index alignment
                for m in reversed(matches[1:]):
                    start, end = m.span()
                    cleaned_text = cleaned_text[:start] + " " + cleaned_text[end:]
                    
    return cleaned_text, penalty_applied, sorted(list(set(flagged_keywords)))

def anonymize_text(text, nltk_available=False):
    """
    AI De-biasing & Anonymization Engine:
    Redacts candidate names, street addresses, emails, phone numbers, and gender pronouns.
    """
    # 1. Redact Emails and Phones
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,13})'
    anonymized = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
    anonymized = re.sub(phone_pattern, "[PHONE_REDACTED]", anonymized)
    
    # 2. Redact Street Addresses
    address_pattern = r'\b\d{1,5}\s+[A-Za-z0-9\s.,#-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Apartment|Apt|Suite|Ste|PO Box)\b'
    anonymized = re.sub(address_pattern, "[ADDRESS_REDACTED]", anonymized, flags=re.IGNORECASE)
    
    # 3. Neutralize Gender Pronouns & Honorifics
    gender_words = {
        r'\bhe\b': 'they', r'\bshe\b': 'they',
        r'\bhis\b': 'their', r'\bher\b': 'their',
        r'\bhim\b': 'them', r'\bhers\b': 'theirs',
        r'\bhimself\b': 'themselves', r'\bherself\b': 'themselves',
        r'\bmale\b': '[GENDER_MASKED]', r'\bfemale\b': '[GENDER_MASKED]',
        r'\bman\b': '[PERSON_MASKED]', r'\bwoman\b': '[PERSON_MASKED]',
        r'\bmr\.\b': '[TITLE_MASKED]', r'\bms\.\b': '[TITLE_MASKED]', r'\bmrs\.\b': '[TITLE_MASKED]',
        r'\bgentleman\b': '[PERSON_MASKED]', r'\blady\b': '[PERSON_MASKED]'
    }
    for pattern, repl in gender_words.items():
        anonymized = re.sub(pattern, repl, anonymized, flags=re.IGNORECASE)
        
    # 4. Redact candidate name headers (first 2 non-empty lines)
    lines = anonymized.split('\n')
    non_empty_line_count = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            non_empty_line_count += 1
            if non_empty_line_count <= 2:
                is_heading = any(h in stripped.lower() for h in ["summary", "profile", "experience", "education", "skills", "objective"])
                if not is_heading and len(stripped) < 60:
                    lines[idx] = "[CANDIDATE_HEADER_REDACTED]"
    anonymized = '\n'.join(lines)
    
    # 5. Redact Proper Nouns (NNPs) using NLTK POS tagger if available
    if nltk_available:
        # Import nltk and modules locally to avoid exceptions if nltk is not loaded yet
        import nltk
        try:
            tokens = nltk.word_tokenize(anonymized)
            pos_tags = nltk.pos_tag(tokens)
            redacted_tokens = []
            skills_lower = set(s.lower() for s in COMMON_SKILLS)
            
            for token, tag in pos_tags:
                if tag in ['NNP', 'NNPS'] and token.lower() not in skills_lower and token.isalpha():
                    redacted_tokens.append("[NNP_REDACTED]")
                else:
                    redacted_tokens.append(token)
            anonymized = " ".join(redacted_tokens)
            anonymized = re.sub(r'\s+([,.:;?!])', r'\1', anonymized)
        except Exception as e:
            logger.warning(f"POS name anonymization failed: {str(e)}")
            
    return anonymized
