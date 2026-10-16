import re

DEGREE_ALIASES = {
    "be": ["be", "b.e", "b.e.", "bachelor of engineering"],
    "btech": ["btech", "b.tech", "b.tech.", "bachelor of technology"],
    "mtech": ["mtech", "m.tech", "m.tech.", "master of technology"],
    "me": ["me", "m.e", "m.e.", "master of engineering"],
    "mca": ["mca", "master of computer applications"],
    "bca": ["bca", "bachelor of computer applications"],
    "bsc": ["bsc", "b.sc", "b.sc.", "bachelor of science"],
    "msc": ["msc", "m.sc", "m.sc.", "master of science"],
    "bachelors": ["bachelor", "bachelors", "bachelor's", "undergraduate"],
    "masters": ["master", "masters", "master's", "postgraduate"]
}

BRANCH_ALIASES = {
    "cs": ["cs", "cse", "computer science", "computer engineering", "computing"],
    "it": ["it", "information technology", "information science", "ise"],
    "ece": ["ece", "electronics and communication", "electronics & communication", "electronics"],
    "eee": ["eee", "electrical and electronics", "electrical & electronics", "electrical"],
    "ai_ml": ["artificial intelligence", "machine learning", "ai", "ml", "ai & ml", "ai/ml", "data science"],
    "mech": ["mechanical", "mechanical engineering", "mech"],
    "civil": ["civil", "civil engineering"]
}


def detect_structured_requirement_type(req_text: str, category: str = "") -> str:
    """
    Determines if a requirement should be evaluated via structured deterministic logic.
    Returns: 'cgpa', 'percentage', 'graduation_year', 'backlog', 'degree', 'branch', 'experience', or None.
    """
    clean = req_text.lower().strip()

    # CGPA
    if re.search(r"\b(cgpa|gpa|cumulative\s+gpa|minimum\s+academic\s+score)\b", clean) and re.search(r"\b\d+(\.\d+)?\b", clean):
        return "cgpa"

    # Percentage
    if re.search(r"\b(\d+%\s*(and\s+above)?|minimum\s*\d+%|\d+\s*percent)\b", clean) and not re.search(r"\bcgpa\b", clean):
        return "percentage"

    # Backlog
    if re.search(r"\b(no\s+active\s+backlogs?|0\s+backlogs?|zero\s+backlogs?|no\s+backlogs?|active\s+backlogs?)\b", clean):
        return "backlog"

    # Graduation Year / Batch
    if re.search(r"\b(20\d{2})\s*(graduating|passed\s*out)?\s*batch\b", clean) or re.search(r"\b(graduating|passout|expected\s+graduation|batch)\s*[:\-]?\s*(20\d{2})\b", clean):
        return "graduation_year"

    # Branch
    if re.search(r"\bbranch\s*[:\-]", clean) or (category == "eligibility" and any(k in clean for k in ["cs", "it", "ece", "eee", "computer science", "branch"])):
        return "branch"

    # Degree
    if category == "education" or re.search(r"\b(degree\s*:|be\s*/|b\.?tech|m\.?tech|mca|bca|bachelor|master)\b", clean):
        return "degree"

    # Experience
    if category == "experience" or re.search(r"\b(\d+\s*-\s*\d+|\d+\+?)\s*years?(\s*of)?(\s*experience)?\b|\bcampus\s+hiring\b|\bfresher\b", clean):
        return "experience"

    return None


def extract_line_with_text(full_text: str, target: str) -> str:
    """Extracts the first non-empty line containing the target snippet."""
    if not target or not full_text:
        return target or ""
    for line in full_text.split("\n"):
        if target.lower() in line.lower():
            cleaned = line.strip()
            if cleaned:
                return cleaned
    return target


def evaluate_cgpa(req_text: str, resume_text: str) -> dict:
    """
    Structured numeric evaluation for CGPA requirements.
    Supports formats: 'CGPA: 8.63', 'CGPA - 8.63', '8.63 CGPA', '8.63/10', 'GPA: 3.8/4.0', 'CGPA of 8.63'.
    """
    # Required CGPA cutoff
    req_match = re.search(r"\b(?:cgpa|gpa|score)?\s*[:\-]?\s*(\d+\.\d+|\d+)\b", req_text, re.I)
    req_cgpa = float(req_match.group(1)) if req_match else 6.0

    # Search for candidate CGPA in resume
    patterns = [
        r"\b(?:cgpa|gpa|cumulative\s+gpa|grade\s+point\s+average)\s*(?:of|[:\-])?\s*(\d+\.\d+)(?:\s*/\s*(10|4)(?:\.0)?)?\b",
        r"\b(\d+\.\d+)\s*(?:/\s*(10|4)(?:\.0)?)?\s*(?:cgpa|gpa)\b",
        r"\b(\d+\.\d+)\s*/\s*(10|4)(?:\.0)?\b"
    ]

    cand_val = None
    cand_scale = 10.0
    matched_snippet = ""

    for pat in patterns:
        m = re.search(pat, resume_text, re.I)
        if m:
            val_str = m.group(1)
            cand_val = float(val_str)
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                cand_scale = float(m.group(2))
            matched_snippet = m.group(0)
            break

    if cand_val is not None:
        line = extract_line_with_text(resume_text, matched_snippet or str(cand_val))

        # Scale normalization if candidate is on a 4.0 scale and requirement is on 10.0 scale
        effective_cgpa = cand_val
        if cand_scale == 4.0 and req_cgpa > 4.0:
            effective_cgpa = round((cand_val / 4.0) * 10.0, 2)

        if effective_cgpa >= req_cgpa:
            return {
                "status": "STRONG EVIDENCE",
                "match_method": "structured_numeric",
                "score": 1.0,
                "candidate_value": cand_val,
                "required_value": req_cgpa,
                "evidence_text": line or f"CGPA: {cand_val}"
            }
        else:
            return {
                "status": "MISSING / UNVERIFIED",
                "match_method": "structured_numeric",
                "score": 0.0,
                "candidate_value": cand_val,
                "required_value": req_cgpa,
                "evidence_text": f"Candidate CGPA ({cand_val}) does not meet required minimum CGPA ({req_cgpa})."
            }

    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_numeric",
        "score": 0.0,
        "candidate_value": None,
        "required_value": req_cgpa,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_percentage(req_text: str, resume_text: str) -> dict:
    """
    Structured numeric evaluation for academic percentage requirements.
    e.g. '60% and above' or 'Minimum 65%'
    """
    req_match = re.search(r"(\d+(?:\.\d+)?)\s*%", req_text)
    req_pct = float(req_match.group(1)) if req_match else 60.0

    cand_match = re.search(r"(\d{2}(?:\.\d+)?)\s*%", resume_text)
    if cand_match:
        cand_pct = float(cand_match.group(1))
        line = extract_line_with_text(resume_text, cand_match.group(0))

        if cand_pct >= req_pct:
            return {
                "status": "STRONG EVIDENCE",
                "match_method": "structured_numeric",
                "score": 1.0,
                "candidate_value": f"{cand_pct}%",
                "required_value": f"{req_pct}%",
                "evidence_text": line or f"Percentage: {cand_pct}%"
            }
        else:
            return {
                "status": "MISSING / UNVERIFIED",
                "match_method": "structured_numeric",
                "score": 0.0,
                "candidate_value": f"{cand_pct}%",
                "required_value": f"{req_pct}%",
                "evidence_text": f"Candidate percentage ({cand_pct}%) does not meet required minimum ({req_pct}%)."
            }

    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_numeric",
        "score": 0.0,
        "candidate_value": None,
        "required_value": f"{req_pct}%",
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_graduation_year(req_text: str, resume_text: str) -> dict:
    """
    Structured eligibility evaluation for Graduation Year / Batch.
    Does NOT infer graduation year from arbitrary dates; requires explicit graduation/completion context.
    """
    req_match = re.search(r"\b(20\d{2})\b", req_text)
    req_year = req_match.group(1) if req_match else None

    if not req_year:
        return None

    # Search for explicit graduation or degree completion indicators
    patterns = [
        r"\b(?:graduating|graduation|batch|passed\s*out|passout|class\s*of|expected\s+graduation)\s*[:\-]?\s*(20\d{2})\b",
        r"\b(?:0[1-9]|1[0-2])[-/](?:20\d{2})\s*-\s*(?:0[1-9]|1[0-2])[-/](20\d{2})\b",
        r"\b(?:20\d{2})\s*-\s*(20\d{2})\b"
    ]

    explicit_cand_year = None
    matched_snippet = ""

    for pat in patterns:
        m = re.search(pat, resume_text, re.I)
        if m:
            explicit_cand_year = m.group(1)
            matched_snippet = m.group(0)
            break

    if explicit_cand_year:
        line = extract_line_with_text(resume_text, matched_snippet or explicit_cand_year)
        if explicit_cand_year == req_year:
            return {
                "status": "STRONG EVIDENCE",
                "match_method": "structured_eligibility",
                "score": 1.0,
                "candidate_value": explicit_cand_year,
                "required_value": req_year,
                "evidence_text": line or f"Graduation Batch: {explicit_cand_year}"
            }
        else:
            return {
                "status": "MISSING / UNVERIFIED",
                "match_method": "structured_eligibility",
                "score": 0.0,
                "candidate_value": explicit_cand_year,
                "required_value": req_year,
                "evidence_text": f"Candidate graduation year ({explicit_cand_year}) does not match required batch ({req_year})."
            }

    # Strict rule: Do not infer graduation years without explicit evidence
    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_eligibility",
        "score": 0.0,
        "candidate_value": None,
        "required_value": req_year,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_backlog(req_text: str, resume_text: str) -> dict:
    """
    Structured eligibility evaluation for Active Backlogs.
    Does NOT assume missing backlogs are zero.
    """
    cand_match = re.search(r"\b(?:active\s+)?backlogs?\s*[:\-]?\s*(\d+|none|zero|nil|no)\b", resume_text, re.I)

    if cand_match:
        val = cand_match.group(1).lower()
        line = extract_line_with_text(resume_text, cand_match.group(0))

        if val in ["0", "none", "zero", "nil", "no"]:
            return {
                "status": "STRONG EVIDENCE",
                "match_method": "structured_eligibility",
                "score": 1.0,
                "candidate_value": 0,
                "required_value": 0,
                "evidence_text": line or "Active Backlogs: 0"
            }
        else:
            return {
                "status": "MISSING / UNVERIFIED",
                "match_method": "structured_eligibility",
                "score": 0.0,
                "candidate_value": val,
                "required_value": 0,
                "evidence_text": line or f"Active Backlogs: {val}"
            }

    # If resume does not explicitly mention backlogs, do NOT assume zero!
    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_eligibility",
        "score": 0.0,
        "candidate_value": None,
        "required_value": 0,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_degree(req_text: str, resume_text: str) -> dict:
    """Structured degree alias matching evaluation."""
    req_clean = req_text.lower()
    res_clean = resume_text.lower()

    allowed_keys = []
    for key, aliases in DEGREE_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", req_clean):
                allowed_keys.append(key)
                break

    if not allowed_keys:
        if "bachelor" in req_clean or "undergraduate" in req_clean:
            allowed_keys.extend(["be", "btech", "bca", "bsc", "bachelors"])
        elif "master" in req_clean or "postgraduate" in req_clean:
            allowed_keys.extend(["mtech", "me", "mca", "msc", "masters"])

    for key in set(allowed_keys):
        aliases = DEGREE_ALIASES.get(key, [key])
        for alias in aliases:
            match = re.search(r"\b" + re.escape(alias) + r"\b", res_clean)
            if match:
                line = extract_line_with_text(resume_text, alias)
                return {
                    "status": "STRONG EVIDENCE",
                    "match_method": "structured_degree",
                    "score": 1.0,
                    "candidate_value": alias.upper(),
                    "required_value": req_text,
                    "evidence_text": line or f"Degree: {alias.upper()}"
                }

    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_degree",
        "score": 0.0,
        "candidate_value": None,
        "required_value": req_text,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_branch(req_text: str, resume_text: str) -> dict:
    """Structured branch / engineering discipline matching evaluation."""
    req_clean = req_text.lower()
    res_clean = resume_text.lower()

    target_branches = []
    for branch_key, aliases in BRANCH_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", req_clean):
                target_branches.append((branch_key, aliases))
                break

    if not target_branches:
        # Check raw words
        words = [w.strip() for w in re.split(r"[/,:\s]+", req_clean) if len(w.strip()) > 1]
        for w in words:
            for b_key, aliases in BRANCH_ALIASES.items():
                if w in aliases:
                    target_branches.append((b_key, aliases))

    for b_key, aliases in target_branches:
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", res_clean):
                line = extract_line_with_text(resume_text, alias)
                return {
                    "status": "STRONG EVIDENCE",
                    "match_method": "structured_branch",
                    "score": 1.0,
                    "candidate_value": alias.title(),
                    "required_value": req_text,
                    "evidence_text": line or f"Branch: {alias.title()}"
                }

    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_branch",
        "score": 0.0,
        "candidate_value": None,
        "required_value": req_text,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_experience(req_text: str, resume_text: str) -> dict:
    """Structured evaluation for explicit experience criteria (e.g. campus hiring, 0-1 years)."""
    clean_req = req_text.lower()
    clean_res = resume_text.lower()

    if "campus hiring" in clean_req or "fresher" in clean_req or "0-1" in clean_req:
        # Look for education or student indicators
        if any(w in clean_res for w in ["bachelor", "b.tech", "be", "master", "student", "graduating", "education"]):
            line = extract_line_with_text(resume_text, "Bachelor") or extract_line_with_text(resume_text, "Education")
            return {
                "status": "STRONG EVIDENCE",
                "match_method": "structured_experience",
                "score": 1.0,
                "candidate_value": "Fresher / Student",
                "required_value": req_text,
                "evidence_text": line or "Eligible for Campus / Fresher hiring"
            }

    exp_match = re.search(r"(\d+)\+?\s*years?", clean_req)
    if exp_match:
        req_years = int(exp_match.group(1))
        cand_exp_match = re.search(r"(\d+)\+?\s*years?\s*(?:of)?\s*experience", clean_res)
        if cand_exp_match:
            cand_years = int(cand_exp_match.group(1))
            line = extract_line_with_text(resume_text, cand_exp_match.group(0))
            if cand_years >= req_years:
                return {
                    "status": "STRONG EVIDENCE",
                    "match_method": "structured_experience",
                    "score": 1.0,
                    "candidate_value": f"{cand_years} years",
                    "required_value": f"{req_years} years",
                    "evidence_text": line
                }
            else:
                return {
                    "status": "MISSING / UNVERIFIED",
                    "match_method": "structured_experience",
                    "score": 0.0,
                    "candidate_value": f"{cand_years} years",
                    "required_value": f"{req_years} years",
                    "evidence_text": f"Candidate experience ({cand_years} years) does not meet requirement ({req_years} years)."
                }

    return {
        "status": "MISSING / UNVERIFIED",
        "match_method": "structured_experience",
        "score": 0.0,
        "candidate_value": None,
        "required_value": req_text,
        "evidence_text": "No strong evidence found in the resume."
    }


def evaluate_structured_requirement(req_item: dict, resume_text: str) -> dict:
    """
    Main entry point for structured evaluation layer.
    Returns result dict if structured matching applies, or None for semantic matching.
    """
    req_text = req_item.get("requirement", "") if isinstance(req_item, dict) else str(req_item)
    category = req_item.get("category", "") if isinstance(req_item, dict) else ""

    req_type = detect_structured_requirement_type(req_text, category)

    if req_type == "cgpa":
        return evaluate_cgpa(req_text, resume_text)
    elif req_type == "percentage":
        return evaluate_percentage(req_text, resume_text)
    elif req_type == "graduation_year":
        return evaluate_graduation_year(req_text, resume_text)
    elif req_type == "backlog":
        return evaluate_backlog(req_text, resume_text)
    elif req_type == "degree":
        return evaluate_degree(req_text, resume_text)
    elif req_type == "branch":
        return evaluate_branch(req_text, resume_text)
    elif req_type == "experience":
        return evaluate_experience(req_text, resume_text)

    return None

