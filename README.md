# Intelligent Candidate Discovery System (Enterprise Security Edition)

A production-ready, bulletproof, highly optimized candidate discovering and ranking system built entirely using local computing resources. The system includes strict parser page boundaries, anti-cheat padding monitors, data anonymization filters, gated scoring matrices, and thread-safe batch loading.

---

## 📐 System Blueprints & Data Flow Map

```
       [Raw Resumes (PDF / DOCX)]        [Job Description Document]
                   │                                  │
                   ▼                                  ▼
       [3-Page Capping Ingestion]            [Hierarchical Ingest]
       - PyMuPDF (fitz) (Pages 1-3)          - PyMuPDF (fitz)
       - pdfplumber (Fallback)               - pdfplumber (Fallback)
       - pypdf (Fallback)                    - pypdf (Fallback)
       - Tesseract OCR (Fallback)            - Tesseract OCR (Fallback)
                   │                                  │
                   ▼                                  ▼
       [Anti-Cheat Padding Scanner]          [AI Anonymization Engine]
       - Scan technical keyword density      - Redact Names & Emails
       - If density > 8% of word count:      - Redact Phone & Addresses
         * Remove duplicate keyword tokens   - Neutralize Gender Markers
         * Schedule 25% final score penalty           │
                   │                                  │
                   ▼                                  │
       [AI Anonymization Engine]                      │
       - Redact Names, Phone, Emails                  │
       - Neutralize Gender Pronouns                   │
                   │                                  │
                   ├──────────────────────────────────┘
                   ▼
         [Skill-Gap Analysis]
      ┌──────────────────────────┐
      │ - Isolated Matched Stack │
      │ - Isolated Missing Gaps  │
      └────────────┬─────────────┘
                   │
                   ▼
       [Local Cached Embeddings]
      ┌──────────────────────────┐
      │ - Hashed clean text key  │
      │ - Check 1024 LRU Cache   │
      │ - Local MiniLM Encoder   │
      │ - Fallback: SK-Learn TFIDF│
      └────────────┬─────────────┘
                   │
                   ▼
         [Gated Math Ranker]
      ┌─────────────────────────────────────────────────────────────┐
      │ 1. Threshold Gate: If Base Cosine Sim < 0.45:               │
      │    - Set Skill Match Weight and Exp Boost to 0              │
      │    - Final Score = 0.60 * Semantic Similarity                │
      │ 2. If Base Cosine Sim >= 0.45:                              │
      │    - Final Score = 0.60*Sem + 0.25*SkillRatio + 0.15*Exp     │
      │ 3. Apply 25% Penalty (Score * 0.75) if padding flag set     │
      └─────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
                [Streamlit Responsive Leaderboard & UI]
      ┌─────────────────────────────────────────────────────────────┐
      │ - Anonymized Toggle View - Diagnostic Pipeline Logs         │
      │ - Green/Red Pill Badges  - Native Match vs Experience Chart │
      └─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Gated Mathematical Scoring Matrix

The system prevents linear score-stuffing vulnerabilities by passing cosine mapping scores through a strict conditional threshold gate:

### 1. Core Profile Gate (Base Similarity Threshold)
$$\text{If Base Semantic Cosine Similarity} < 0.45:$$

$$\text{Skill Match Ratio} = 0.0$$
$$\text{Experience Match Weight} = 0.0$$
$$\text{Final Score} = 0.60 \times \text{Semantic Similarity}$$

---

### 2. Standard Scoring Weight Formula
$$\text{If Base Semantic Cosine Similarity} \ge 0.45:$$

$$\text{Final Score} = (0.60 \times \text{Semantic Similarity}) + (0.25 \times \text{Skill Match Ratio}) + (0.15 \times \text{Experience Match Weight})$$

Where:
-   **Skill Match Ratio**: $\frac{|\text{Candidate Skills} \cap \text{JD Skills}|}{|\text{JD Skills}|}$
-   **Experience Match Weight**: $\min\left(1.0, \frac{E_{cand}}{E_{req}}\right)$ *(Defaults to $1.0$ if $E_{req} = 0$).*

---

### 3. Anti-Cheat Padding Penalty (Density Enforcement)
Let $C_k$ represent the count of keyword $k$ in the resume text and $W$ represent the total word count. If any single technical keyword density exceeds 8%:

$$\exists k \in K \text{ such that } \frac{C_k}{W} > 0.08 \implies \text{Padding Flag} = \text{True}$$

$$\text{Final Score} = \text{Final Score} \times 0.75 \quad \text{(Applied 25% relative penalty)}$$

---

## ⚡ Key Architectural Safeguards

-   **Memory Leak Safety**: Triggers garbage collection routines (`gc.collect()`) after batch-processing runs to clear active tensor vectors and prevent memory overflow crashes on batch multi-uploads (20+ files).
-   **3-Page PDF Cap Boundaries**: Forces parsers to truncate processing at exactly page 3. This guarantees OOM safety against maliciously large/recursive document attachments.
-   **Anonymous View Sourcing**: A toggle in the Streamlit Sidebar hides Names, Emails, and Phone Numbers in expanded detail views, displaying them instead as standard hashes (e.g. `Candidate_001`), ensuring unbiased screening.

---

## 🚀 Local Deployment Guide

1.  Clone this folder and navigate into it:
    ```bash
    cd Intelligent-Candidate-Discovery
    ```
2.  Install pinned open-source dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
