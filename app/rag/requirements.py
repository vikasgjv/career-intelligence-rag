import re

# Requirement Categories
CAT_TECH = "technical_skill"
CAT_RESP = "responsibility"
CAT_EDU = "education"
CAT_ELIG = "eligibility"
CAT_EXP = "experience"
CAT_SOFT = "soft_skill"
CAT_DOMAIN = "domain_knowledge"

COMPANY_SECTION_PATTERNS = [
    r"^about\s+(the\s+)?(company|team|organization|us|[a-z0-9_\-]+)",
    r"^who\s+we\s+are",
    r"^our\s+(company|culture|mission|story|values|vision|team)",
    r"^company\s+(overview|description|profile|background)",
    r"^why\s+([a-z0-9_\-]+|join\s+us|work\s+with\s+us)",
    r"^benefits(\s+and\s+perks)?",
    r"^perks(\s+and\s+benefits)?",
    r"^what\s+we\s+offer",
    r"^compensation(\s+and\s+benefits)?",
    r"^equal\s+opportunity(\s+employer)?",
    r"^diversity(\s+and\s+inclusion)?",
    r"^eeo(\s+statement)?",
    r"^corporate\s+information",
    r"^how\s+to\s+apply",
    r"^salary(\s+range)?",
    r"^location(s)?$"
]

REQUIREMENT_SECTION_PATTERNS = [
    r"^job\s+(description|summary)$",
    r"^(your\s+)?role(\s+and\s+responsibilities)?$",
    r"^responsibilities$",
    r"^key\s+responsibilities$",
    r"^what\s+you('ll|\s+will)\s+do$",
    r"^(required|preferred)?\s*(professional\s+and\s+technical\s+)?expertise$",
    r"^(required|preferred|basic)?\s*skills$",
    r"^(technical\s+)?skills\s*(required)?$",
    r"^tech\s+stack$",
    r"^(required|preferred|basic)?\s*qualifications$",
    r"^eligibility(\s+criteria)?$",
    r"^educational\s+qualifications$",
    r"^education$",
    r"^academic\s+criteria$",
    r"^experience(\s+required)?$",
    r"^what\s+you\s+need$",
    r"^what\s+we\s+are\s+looking\s+for$",
    r"^requirements$"
]

# Universal boilerplate and marketing phrases (applicable to any employer)
GENERAL_BOILERPLATE_PATTERNS = [
    r"equal\s+opportunity\s+employer",
    r"all\s+qualified\s+applicants(\s+will\s+receive\s+consideration)?",
    r"without\s+regard\s+to\s+(race|color|religion|sex|sexual\s+orientation)",
    r"we\s+celebrate\s+diversity",
    r"affirmative\s+action",
    r"founded\s+in\s+\d{4}",
    r"headquartered\s+in",
    r"apply\s+online",
    r"competitive\s+compensation",
    r"comprehensive\s+benefits\s+package",
    r"health,\s+dental,\s+and\s+vision",
    r"paid\s+time\s+off",
    r"unlimited\s+pto",
    r"click\s+here\s+to\s+apply",
    r"how\s+to\s+apply",
    r"salary\s+range\s*[:\-]",
    r"^posting\s+id\s*[:\-]",
    r"^industry\s+type\s*[:\-]"
]

HEADER_METADATA_PATTERNS = [
    r"^job\s+title\s*[:\-]",
    r"^role\s*[:\-]",
    r"^locations?\s*[:\-]",
    r"^posting\s+id\s*[:\-]",
    r"^employment\s+type\s*[:\-]",
    r"^www\.[a-z0-9\-\.]+\.[a-z]{2,}",
    r"^http[s]?://",
    r"^what\s+do\s+we\s+offer"
]

KNOWN_TECH_SKILLS = [
    "Python", "Java", "C++", "C#", "SQL", "JavaScript", "TypeScript",
    "Node.js", "React", "Angular", "Vue.js", "Vue", "FastAPI", "Flask", "Django",
    "Express", "HTML", "CSS", "Git", "GitHub", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "TensorFlow", "PyTorch", "Scikit-learn",
    "Pandas", "NumPy", "Machine Learning", "Artificial Intelligence", "AI",
    "Deep Learning", "Natural Language Processing", "NLP", "Computer Vision",
    "REST API", "REST APIs", "Data Structures", "Algorithms",
    "Data Structures and Algorithms", "Database Management", "Microservices",
    "CI/CD", "Linux", "Unix", "Blockchain", "Quantum Computing", "Programming"
]

SOFT_SKILL_KEYWORDS = {
    "problem solving": "Problem Solving",
    "communication skills": "Communication Skills",
    "interpersonal skills": "Interpersonal Skills",
    "stakeholder management": "Stakeholder Management",
    "team collaboration": "Team Collaboration",
    "analytical thinking": "Analytical Thinking",
    "leadership": "Leadership",
    "time management": "Time Management",
    "critical thinking": "Critical Thinking"
}


def is_company_section_header(line: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", line.strip().lower())
    for pat in COMPANY_SECTION_PATTERNS:
        if re.search(pat, clean):
            return True
    return False


def is_requirement_section_header(line: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", line.strip().lower())
    for pat in REQUIREMENT_SECTION_PATTERNS:
        if re.search(pat, clean):
            return True
    return line.rstrip(":").strip().lower() in [
        "responsibilities", "qualifications", "requirements", "eligibility", "skills", "experience"
    ] or (line.endswith(":") and len(line.split()) <= 4)


def is_marketing_or_metadata(text: str) -> bool:
    text_lower = text.strip().lower()

    for pat in HEADER_METADATA_PATTERNS:
        if re.search(pat, text_lower):
            return True

    for pat in GENERAL_BOILERPLATE_PATTERNS:
        if re.search(pat, text_lower):
            return True

    # Generic company background descriptions like "XYZ is a global provider of..."
    if re.search(r"\b(about\s+[a-z0-9]+|[a-z0-9]+\s+is\s+(one|a|our|the))\b", text_lower):
        if not any(k in text_lower for k in ["python", "java", "c++", "degree", "cgpa", "batch", "backlogs", "sql"]):
            return True

    return False


def clean_bullet(text: str) -> str:
    """Removes bullet symbols and leading enumerators."""
    text = re.sub(r"^(?:[•·▪\-*\u2022\u2023\u2043\u204c\u204d\u2219]+|\(?\d+[\.\)]|\([a-zA-Z]\))\s*", "", text).strip()
    text = re.sub(r"^(job summary|responsibilities|qualifications|eligibility|requirements|skills)[:\s]*", "", text, flags=re.I).strip()
    text = text.strip().rstrip(".,;")
    return text


def parse_jd_into_sections(jd_text: str):
    """Splits job description text into named sections."""
    lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
    sections = []
    current_title = "General"
    current_lines = []

    for line in lines:
        if is_company_section_header(line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        elif is_requirement_section_header(line) and len(line.split()) <= 6:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return sections


def find_tech_skills_in_text(text: str) -> list[str]:
    """Finds all distinct known technical skills in a line of text."""
    found = []
    for tech in KNOWN_TECH_SKILLS:
        if tech == "C++":
            pat = r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9\+])"
        elif tech == "C#":
            pat = r"(?<![a-zA-Z0-9])c#(?![a-zA-Z0-9#])"
        elif tech == "C":
            pat = r"\bc\b(?![+#])"
        elif tech in ["AI", "NLP", "SQL", "HTML", "CSS", "AWS", "GCP", "CI/CD"]:
            pat = r"\b" + re.escape(tech) + r"\b"
        else:
            pat = r"\b" + re.escape(tech) + r"\b"

        if re.search(pat, text, re.I):
            found.append(tech)
    return found



def extract_requirements(job_description: str) -> list[dict]:
    """
    Section-aware Job Description requirement extraction pipeline.
    Parses sections, strips company boilerplate, splits combined requirements into
    independently matchable items, and preserves real source sentences for traceability.
    """
    if not job_description:
        return []

    sections = parse_jd_into_sections(job_description)
    raw_requirements = []

    for sec_title, sec_lines in sections:
        if is_company_section_header(sec_title):
            continue

        for line in sec_lines:
            cleaned = clean_bullet(line)
            if not cleaned or len(cleaned) < 3:
                continue

            if is_marketing_or_metadata(cleaned):
                continue

            c_lower = cleaned.lower()
            line_decomposed = False

            # 1. Batch / Graduation Year Requirement
            batch_match = re.search(r"\b(\d{4}\s*(graduating|passed\s*out)?\s*batch\s*(only)?)\b", cleaned, re.I)
            if batch_match:
                raw_requirements.append({
                    "requirement": cleaned if len(cleaned) < 60 else batch_match.group(1).strip(),
                    "category": CAT_ELIG,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True

            # 2. Education Degrees
            deg_match = re.search(r"\b(Degree\s*:[^\.\n]+|BE|B\.?Tech|ME|M\.?Tech|MCA|BCA|B\.?Sc|M\.?Sc|Bachelor'?s|Master'?s|PhD)\b", cleaned, re.I)
            if deg_match:
                deg_str = cleaned if ("degree" in c_lower or len(cleaned) < 50) else deg_match.group(0).strip()
                raw_requirements.append({
                    "requirement": deg_str,
                    "category": CAT_EDU,
                    "source_section": sec_title,
                    "source_text": line
                })

            # 3. Branch Criteria
            branch_match = re.search(r"\b(branch\s*:[^\.\n]+)\b", cleaned, re.I)
            if branch_match:
                raw_requirements.append({
                    "requirement": branch_match.group(1).strip(),
                    "category": CAT_ELIG,
                    "source_section": sec_title,
                    "source_text": line
                })

            # 4. Academic Score / Cutoff (CGPA or Percentage)
            cgpa_match = re.search(r"\b(cgpa\s*[\d\.]+|minimum\s*academic\s*score[^\.\n]+|\d+%\s*and\s*above|minimum\s*\d+%)\b", cleaned, re.I)
            if cgpa_match and not deg_match and not branch_match:
                raw_requirements.append({
                    "requirement": cleaned if len(cleaned) < 70 else cgpa_match.group(0).strip(),
                    "category": CAT_ELIG,
                    "source_section": sec_title,
                    "source_text": line
                })

            # 5. Active Backlogs
            if re.search(r"\b(no\s+active\s+backlogs|0\s+backlogs|zero\s+backlogs)\b", c_lower):
                raw_requirements.append({
                    "requirement": "No Active Backlogs",
                    "category": CAT_ELIG,
                    "source_section": sec_title,
                    "source_text": line
                })

            # 6. Experience
            exp_match = re.search(r"\b(\d+\s*-\s*\d+|\d+\+?)\s*years?(\s*of)?(\s*experience)?\b|\bcampus\s+hiring\b|\bfresher\b", c_lower)
            if exp_match and not batch_match and not branch_match and not deg_match:
                raw_requirements.append({
                    "requirement": cleaned if len(cleaned) < 60 else exp_match.group(0).strip(),
                    "category": CAT_EXP,
                    "source_section": sec_title,
                    "source_text": line
                })

            # 7. Soft Skills
            for kw, label in SOFT_SKILL_KEYWORDS.items():
                if kw in c_lower:
                    raw_requirements.append({
                        "requirement": label,
                        "category": CAT_SOFT,
                        "source_section": sec_title,
                        "source_text": line
                    })

            # 8. Splitting Combined Technical Skills in a sentence or list
            # Handles "Programming in Java, C++, Python, or Node.js", "Git and GitHub", "Python, SQL, Machine Learning"
            matched_techs = find_tech_skills_in_text(cleaned)
            is_combined_skill_list = len(matched_techs) > 1 or any(kw in c_lower for kw in [
                "experience in:", "experience with", "proficiency in", "skills in",
                "preferably in", "such as", "including", "programming in", "technologies:"
            ])

            if matched_techs and (is_combined_skill_list or len(cleaned) <= 40):
                for tech in matched_techs:
                    raw_requirements.append({
                        "requirement": tech,
                        "category": CAT_TECH,
                        "source_section": sec_title,
                        "source_text": line
                    })
                line_decomposed = True

            # 9. Capability & Responsibility decomposition
            if "development lifecycle" in c_lower or "sdlc" in c_lower:
                raw_requirements.append({
                    "requirement": "Software Development Life Cycle Concepts",
                    "category": CAT_RESP,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True
            if "design" in c_lower and any(w in c_lower for w in ["software", "system", "code", "architecture"]):
                raw_requirements.append({
                    "requirement": "Software Design",
                    "category": CAT_RESP,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True
            if "testing" in c_lower and any(w in c_lower for w in ["software", "code", "unit", "system", "application"]):
                raw_requirements.append({
                    "requirement": "Software Testing",
                    "category": CAT_RESP,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True
            if ("debug" in c_lower or "failures" in c_lower or "issue" in c_lower) and any(w in c_lower for w in ["analyze", "resolve", "troubleshoot"]):
                raw_requirements.append({
                    "requirement": "Debugging & Issue Analysis",
                    "category": CAT_RESP,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True
            if "propose improvements" in c_lower or "continuous improvement" in c_lower or "solution improvement" in c_lower:
                raw_requirements.append({
                    "requirement": "Solution Improvement",
                    "category": CAT_RESP,
                    "source_section": sec_title,
                    "source_text": line
                })
                line_decomposed = True

            # 10. Action-oriented Responsibilities
            action_verbs = ["develop", "build", "design", "implement", "maintain", "create", "participate", "collaborate", "lead", "manage", "test", "deploy", "integrate", "analyze", "work with", "use", "write"]
            starts_with_action = any(c_lower.startswith(v) for v in action_verbs)

            if not line_decomposed:
                if (starts_with_action or any(term in sec_title.lower() for term in ["responsibilities", "role", "what you'll do"])) and len(cleaned) <= 120:
                    cat = CAT_RESP if starts_with_action else CAT_TECH
                    raw_requirements.append({
                        "requirement": cleaned,
                        "category": cat,
                        "source_section": sec_title,
                        "source_text": line
                    })
                elif len(cleaned) <= 50 and not is_requirement_section_header(cleaned):
                    # Single short skill / item bullet
                    cat = CAT_TECH if matched_techs else CAT_RESP
                    raw_requirements.append({
                        "requirement": cleaned,
                        "category": cat,
                        "source_section": sec_title,
                        "source_text": line
                    })

    # Deduplicate while preserving source traceability
    seen = set()
    final = []
    for req in raw_requirements:
        req_name = req["requirement"].strip().rstrip(".,;")
        req_lower = req_name.lower()

        if req_lower in seen:
            continue
        if is_marketing_or_metadata(req_name):
            continue
        if is_company_section_header(req_name) or is_requirement_section_header(req_name):
            continue

        seen.add(req_lower)
        req_copy = dict(req)
        req_copy["requirement"] = req_name
        final.append(req_copy)

    return final