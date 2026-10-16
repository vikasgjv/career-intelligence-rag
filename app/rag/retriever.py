import re
import numpy as np
from .config import (
    STRONG_EVIDENCE_THRESHOLD,
    PARTIAL_EVIDENCE_THRESHOLD,
    WEIGHT_STRONG_EVIDENCE,
    WEIGHT_PARTIAL_EVIDENCE,
    WEIGHT_MISSING_EVIDENCE
)
from .structured_matcher import evaluate_structured_requirement
from .hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def extract_exact_evidence_line(chunk_text: str, req_text: str) -> str:
    """
    Isolates the most specific line or sentence in chunk_text supporting req_text.
    Preserves original resume wording without paraphrasing.
    """
    if not chunk_text:
        return ""

    lines = [l.strip() for l in chunk_text.split("\n") if l.strip()]
    if len(lines) <= 1:
        return chunk_text.strip()

    # Check for direct phrase match first
    req_clean = req_text.lower()
    for line in lines:
        if req_clean in line.lower():
            return line

    req_words = [w.lower() for w in re.findall(r"\b\w+\b", req_text) if len(w) > 2]
    if not req_words:
        return lines[0]

    # Score each line by keyword overlap with requirement
    best_line = lines[0]
    best_overlap = -1

    for line in lines:
        line_lower = line.lower()
        overlap = sum(1 for w in req_words if w in line_lower)
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line

    return best_line


def retrieve(query_embedding, embeddings, chunks, top_k=2):
    """
    Computes cosine similarity between query embedding and candidate chunk embeddings.
    Returns top_k matching chunks with similarity scores and rich metadata.
    """
    if len(chunks) == 0 or len(embeddings) == 0:
        return []

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    embeddings_norm = embeddings / (np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    ) + 1e-9)

    similarities = np.dot(embeddings_norm, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for index in top_indices:
        results.append({
            "text": chunks[index]["text"],
            "source": chunks[index].get("source", "resume"),
            "section": chunks[index].get("section", "General"),
            "page": chunks[index].get("page", 1),
            "score": float(similarities[index]),
            "retrieval_method": "semantic_dense"
        })

    return results


def find_page_for_evidence(evidence_text: str, chunks: list[dict]) -> int:
    """Finds which page number contained the evidence snippet."""
    if not evidence_text or not chunks:
        return 1
    clean_ev = evidence_text.lower().strip()
    for c in chunks:
        if clean_ev in c.get("text", "").lower():
            return c.get("page", 1)
    return 1


def match_requirements(requirements, embeddings, chunks, model, top_k=1, full_resume_text=""):
    """
    Matches each JD requirement against resume evidence chunks.
    Routes factual requirements through structured evaluation (CGPA, degree, graduation year, backlogs)
    and hybrid retrieval (BM25 + Dense Semantic search combined via Reciprocal Rank Fusion) for skills and capabilities.
    """
    matches = []

    if not full_resume_text and chunks:
        full_resume_text = "\n".join([c["text"] for c in chunks])

    hybrid_retriever = HybridRetriever(chunks, embeddings=embeddings, model=model) if chunks else None

    for item in requirements:
        if isinstance(item, dict):
            req_text = item["requirement"]
            category = item.get("category", "technical_skill")
            source_section = item.get("source_section", "General")
            source_text = item.get("source_text", req_text)
        else:
            req_text = str(item)
            category = "technical_skill"
            source_section = "General"
            source_text = req_text

        # 1. Structured requirement evaluation layer
        structured_res = evaluate_structured_requirement(
            {"requirement": req_text, "category": category},
            full_resume_text
        )

        if structured_res:
            page_num = find_page_for_evidence(structured_res.get("evidence_text", ""), chunks)
            method = structured_res.get("match_method", "structured_eligibility")
            matches.append({
                "requirement": req_text,
                "category": category,
                "source_section": source_section,
                "source_text": source_text,
                "status": structured_res["status"],
                "score": structured_res["score"],
                "match_method": method,
                "candidate_value": structured_res.get("candidate_value"),
                "required_value": structured_res.get("required_value"),
                "evidence": [{
                    "text": structured_res["evidence_text"],
                    "source": "Resume Structured Evaluation",
                    "section": "Education / Eligibility",
                    "page": page_num,
                    "score": structured_res["score"],
                    "retrieval_method": method
                }]
            })
            continue

        # 2. Hybrid RAG retrieval layer (Dense Semantic + BM25 via Reciprocal Rank Fusion)
        if hybrid_retriever:
            results = hybrid_retriever.retrieve_hybrid(req_text, top_k=top_k)
        else:
            results = []

        best_result = results[0] if results else None
        if best_result:
            dense_score = best_result.get("dense_score", 0.0)
            bm25_score = best_result.get("bm25_score", 0.0)
            retrieval_method = best_result.get("retrieval_method", "hybrid_rrf")

            # Check if exact skill/keyword exists in candidate chunk
            req_lower = req_text.lower()
            exact_keyword_in_chunk = req_lower in best_result["text"].lower()

            if dense_score >= STRONG_EVIDENCE_THRESHOLD or (exact_keyword_in_chunk and dense_score >= 0.25):
                status = "STRONG EVIDENCE"
                confidence_score = max(dense_score, STRONG_EVIDENCE_THRESHOLD if exact_keyword_in_chunk else dense_score)
            elif dense_score >= PARTIAL_EVIDENCE_THRESHOLD or bm25_score > 0.0:
                status = "PARTIAL / RELATED"
                confidence_score = max(dense_score, PARTIAL_EVIDENCE_THRESHOLD)
            else:
                status = "MISSING / UNVERIFIED"
                confidence_score = dense_score

            if status != "MISSING / UNVERIFIED":
                evidence_line = extract_exact_evidence_line(best_result["text"], req_text)
                evidence_item = {
                    "text": evidence_line,
                    "source": best_result["source"],
                    "section": best_result["section"],
                    "page": best_result.get("page", 1),
                    "score": confidence_score,
                    "rrf_score": best_result.get("rrf_score", 0.0),
                    "retrieval_method": retrieval_method
                }
            else:
                evidence_item = {
                    "text": "No strong evidence found in the resume.",
                    "source": "N/A",
                    "section": "N/A",
                    "page": "N/A",
                    "score": confidence_score,
                    "retrieval_method": "none"
                }
        else:
            status = "MISSING / UNVERIFIED"
            confidence_score = 0.0
            evidence_item = {
                "text": "No strong evidence found in the resume.",
                "source": "N/A",
                "section": "N/A",
                "page": "N/A",
                "score": 0.0,
                "retrieval_method": "none"
            }

        matches.append({
            "requirement": req_text,
            "category": category,
            "source_section": source_section,
            "source_text": source_text,
            "status": status,
            "score": confidence_score,
            "match_method": evidence_item["retrieval_method"],
            "evidence": [evidence_item]
        })

    return matches



def categorize_matches(requirement_matches: list[dict]) -> dict:
    """
    Categorizes matches into strong, partial, and missing evidence lists,
    and calculates an evidence-weighted Overall Match Coverage score.
    """
    strong_skills = []
    partial_skills = []
    missing_skills = []

    total_weight = 0.0

    for match in requirement_matches:
        status = match["status"]
        score = match["score"]
        evidence = match["evidence"][0] if match["evidence"] else None

        item = {
            "requirement": match["requirement"],
            "category": match.get("category", "technical_skill"),
            "source_section": match.get("source_section", "General"),
            "source_text": match.get("source_text", match["requirement"]),
            "status": status,
            "score": score,
            "match_method": match.get("match_method", "semantic_dense"),
            "candidate_value": match.get("candidate_value"),
            "required_value": match.get("required_value"),
            "section": evidence["section"] if evidence else "N/A",
            "page": evidence.get("page", 1) if evidence else "N/A",
            "evidence_text": evidence["text"] if evidence else "No strong evidence found in the resume."
        }

        if status == "STRONG EVIDENCE":
            strong_skills.append(item)
            total_weight += WEIGHT_STRONG_EVIDENCE
        elif status == "PARTIAL / RELATED":
            partial_skills.append(item)
            total_weight += WEIGHT_PARTIAL_EVIDENCE
        else:
            missing_skills.append(item)
            total_weight += WEIGHT_MISSING_EVIDENCE

    total_requirements = len(requirement_matches)
    coverage_score = (total_weight / total_requirements * 100) if total_requirements > 0 else 0.0

    return {
        "strong_skills": strong_skills,
        "partial_skills": partial_skills,
        "missing_skills": missing_skills,
        "total_requirements": total_requirements,
        "strong_count": len(strong_skills),
        "partial_count": len(partial_skills),
        "missing_count": len(missing_skills),
        "match_percentage": round(coverage_score, 1)
    }