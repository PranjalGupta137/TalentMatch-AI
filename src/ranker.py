import logging
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_cosine_similarity_sklearn(vec1, vec2):
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])

def calculate_cosine_similarity_numpy(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def calculate_cosine_similarity_manual(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    norm_v1 = math.sqrt(sum(x * x for x in vec1))
    norm_v2 = math.sqrt(sum(x * x for x in vec2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def calculate_manhattan_distance_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")
    manhattan_dist = sum(abs(x - y) for x, y in zip(vec1, vec2))
    return 1.0 / (1.0 + manhattan_dist)

def get_similarity_score(vec1, vec2):
    try:
        return calculate_cosine_similarity_sklearn(vec1, vec2), "cosine-sklearn"
    except Exception as e:
        logger.error(f"Sklearn similarity failed: {str(e)}. Trying NumPy...")
    try:
        return calculate_cosine_similarity_numpy(vec1, vec2), "cosine-numpy"
    except Exception as e:
        logger.error(f"NumPy similarity failed: {str(e)}. Trying manual...")
    try:
        return calculate_cosine_similarity_manual(vec1, vec2), "cosine-manual"
    except Exception as e:
        logger.error(f"Manual similarity failed: {str(e)}. Trying Manhattan distance...")
    try:
        return calculate_manhattan_distance_similarity(vec1, vec2), "manhattan-similarity"
    except Exception as e:
        logger.error(f"Manhattan similarity failed: {str(e)}")
        raise RuntimeError(f"All similarity computations failed: {str(e)}")

def generate_feedback(base_score, skill_ratio, exp_weight, matched_skills, missing_skills, cand_exp, req_exp, is_gated, padding_penalty, flagged_kws, is_resume=True, resume_validation_reason="", gate_threshold=0.30, project_score=0.0, project_details=None):
    """Generates detailed explainable feedback including security warnings and threshold gates."""
    reasons = []
    
    # 0. Resume Validity Alert
    if not is_resume:
        reasons.append(f"Invalid Document: This file is not recognized as a valid professional resume ({resume_validation_reason}). Match score forced to 0%.")
    # 1. Gated Mismatch Alert
    elif is_gated:
        reasons.append(f"Gated Profile: Core semantic compatibility ({base_score:.3f}) is below target threshold ({gate_threshold:.2f}). Skill and experience weights were not added to final score.")
    else:
        # Semantic evaluation
        if base_score >= 0.60:
            reasons.append("Strong semantic alignment with core job responsibilities.")
        elif base_score >= 0.40:
            reasons.append("Good semantic match to target requirements.")
        else:
            reasons.append("Basic semantic correlation with the target role description.")
            
        # Experience evaluation
        if req_exp > 0:
            if cand_exp >= req_exp:
                reasons.append(f"Meets or exceeds experience requirements ({cand_exp} years vs {req_exp} required).")
            else:
                reasons.append(f"Has an experience deficit ({cand_exp} years vs {req_exp} required).")
        else:
            reasons.append(f"Has {cand_exp} years of relevant experience.")
            
        # Skill match evaluation
        if skill_ratio == 1.0:
            reasons.append("Matches 100% of required technical stack.")
        elif skill_ratio >= 0.60:
            reasons.append(f"Matches {len(matched_skills)} core skills. Missing gaps: {', '.join(missing_skills[:3])}.")
        else:
            if missing_skills:
                reasons.append(f"Significant skill gaps. Missing: {', '.join(missing_skills[:4])}.")
            else:
                reasons.append("No overlapping technical skills identified.")
                
        # Project match evaluation
        if project_details:
            cnt = project_details.get("project_count", 0)
            int_skills = project_details.get("integrated_skills", [])
            if project_score >= 0.70:
                reasons.append(f"Demonstrates high-quality projects ({cnt} found) integrating core required tools ({', '.join(int_skills[:3])}) with strong architectural complexity.")
            elif project_score >= 0.40:
                reasons.append(f"Includes relevant project work ({cnt} projects) implementing required skills like {', '.join(int_skills[:2]) if int_skills else 'technologies'}.")
            else:
                reasons.append("Contains limited or basic project documentation in target technologies.")
                
    # 2. Security Warnings
    if padding_penalty:
        reasons.append(f"WARNING: Suspected Resume Padding Violation detected (Excessive keyword stuffing of: {', '.join(flagged_kws)}). A 25% score penalty has been applied.")
        
    return " ".join(reasons)

def rank_candidates(jd_embedding, candidate_embeddings, candidate_metadatas, jd_metadata, gate_threshold=0.30):
    """
    Ranks candidates using gated mathematical scoring rules:
    - If base similarity < gate_threshold: Gated/Mismatched Profile, experience, skill, and project weights set to 0.
    - If base similarity >= gate_threshold: Score = 0.50*Semantic + 0.20*SkillRatio + 0.15*ExpWeight + 0.15*ProjectScore.
    - If padding_penalty_applied is True: Multiply final score by 0.75 (25% penalty).
    """
    ranked_list = []
    required_exp = jd_metadata.get("experience", 0)
    jd_skills = set(jd_metadata.get("skills", []))
    
    for idx, (cand_emb, cand_meta) in enumerate(zip(candidate_embeddings, candidate_metadatas)):
        base_score, method_used = get_similarity_score(jd_embedding, cand_emb)
        
        # Skill Match Ratio
        cand_skills = set(cand_meta.get("skills", []))
        matched_skills = cand_meta.get("matched_skills", sorted(list(jd_skills.intersection(cand_skills))))
        missing_skills = cand_meta.get("missing_skills", sorted(list(jd_skills.difference(cand_skills))))
        
        # Experience Weight
        cand_exp = cand_meta.get("experience", 0)
        
        # Project Score
        project_score = cand_meta.get("project_score", 0.0)
        project_details = cand_meta.get("project_details", {"project_count": 0, "integrated_skills": [], "complexity_score": 0.0})
        
        is_resume = cand_meta.get("is_resume", True)
        resume_validation_reason = cand_meta.get("resume_validation_reason", "")
        
        # Check Gate Threshold
        is_gated = (base_score < gate_threshold) or (not is_resume)
        
        if not is_resume:
            skill_ratio = 0.0
            exp_weight = 0.0
            p_score = 0.0
            final_score = 0.0
        elif is_gated:
            skill_ratio = 0.0
            exp_weight = 0.0
            p_score = 0.0
            # Force linear weights to 0, score depends solely on 0.60 * similarity
            final_score = 0.60 * base_score
        else:
            skill_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 1.0
            if required_exp > 0:
                exp_weight = min(1.0, cand_exp / required_exp)
            else:
                exp_weight = 1.0
            p_score = project_score
            final_score = (0.50 * base_score) + (0.20 * skill_ratio) + (0.15 * exp_weight) + (0.15 * p_score)
            
        # Check Security Padding Penalty
        padding_penalty = cand_meta.get("padding_penalty_applied", False)
        flagged_kws = cand_meta.get("flagged_keywords", [])
        
        if padding_penalty:
            # Apply 25% score reduction
            final_score = final_score * 0.75
            
        final_score = max(0.0, min(1.0, final_score))
        
        feedback = generate_feedback(
            base_score, skill_ratio, exp_weight, 
            matched_skills, missing_skills, cand_exp, required_exp, 
            is_gated, padding_penalty, flagged_kws,
            is_resume=is_resume, resume_validation_reason=resume_validation_reason,
            gate_threshold=gate_threshold,
            project_score=p_score, project_details=project_details
        )
        
        ranked_list.append({
            "candidate_index": idx,
            "name": cand_meta.get("name", f"Candidate_{idx+1}"),
            "base_score": base_score,
            "skill_ratio": skill_ratio,
            "exp_weight": exp_weight,
            "project_score": p_score,
            "project_details": project_details,
            "final_score": final_score,
            "match_percentage": f"{final_score * 100:.1f}%",
            "experience": cand_exp,
            "skills": cand_meta.get("skills", []),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "is_gated": is_gated,
            "is_resume": is_resume,
            "resume_validation_reason": resume_validation_reason,
            "padding_penalty_applied": padding_penalty,
            "flagged_keywords": flagged_kws,
            "feedback": feedback,
            "similarity_method": method_used,
            "email": cand_meta.get("email", "N/A"),
            "phone": cand_meta.get("phone", "N/A")
        })
        
    ranked_list = sorted(ranked_list, key=lambda x: x["final_score"], reverse=True)
    return ranked_list

def generate_explainable_dataframe(ranked_results):
    """
    Converts ranked results into structured Pandas DataFrame.
    """
    data = []
    for rank, res in enumerate(ranked_results):
        status = "Passed Gate"
        if not res.get("is_resume", True):
            status = "Invalid Document"
        elif res["padding_penalty_applied"]:
            status = "Security Flag (Padding)"
        elif res["is_gated"]:
            status = "Gated (Low Alignment)"
            
        data.append({
            "Rank": rank + 1,
            "Candidate Name": res["name"],
            "Match Score (%)": res["match_percentage"],
            "Base Semantic Score": round(res["base_score"], 3),
            "Skill Matches": f"{len(res['matched_skills'])} / {len(res['matched_skills']) + len(res['missing_skills'])}",
            "Project Score (%)": f"{res['project_score'] * 100:.1f}%",
            "Yrs Experience": res["experience"],
            "Security Status": status,
            "Feedback": res["feedback"]
        })
    return pd.DataFrame(data)
