import re

SECTION_PATTERNS = [
    (r"\b(technical\s+skills|skills|core\s+competencies|tech\s+stack)\b", "Skills"),
    (r"\b(education|academic\s+background|qualifications|academics)\b", "Education"),
    (r"\b(projects|key\s+projects|academic\s+projects)\b", "Projects"),
    (r"\b(work\s+experience|experience|professional\s+experience|employment)\b", "Experience"),
    (r"\b(profile\s+summary|summary|profile|objective|about\s+me)\b", "Profile Summary"),
    (r"\b(certifications|licenses|awards|achievements)\b", "Certifications"),
    (r"\b(languages|interests|hobbies)\b", "Languages & Interests")
]

SUBSECTION_PREFIXES = [
    "languages", "tools", "frameworks", "soft skills", "core areas",
    "databases", "cloud", "platforms"
]


def _chunk_single_page(text: str, source: str = "resume", page_number: int = 1) -> list[dict]:
    """Internal helper to split text from a single page into section-aware chunks."""
    if not text:
        return []

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 1. Group lines into top-level sections
    sections = []
    current_sec_name = "General"
    current_sec_lines = []

    for line in lines:
        detected_sec = None
        line_clean = line.rstrip(":").strip().lower()

        for pat, sec_name in SECTION_PATTERNS:
            if re.search(pat, line_clean, re.I) and len(line_clean.split()) <= 5:
                detected_sec = sec_name
                break
        if detected_sec:
            if current_sec_lines:
                sections.append((current_sec_name, current_sec_lines))
            current_sec_name = detected_sec
            current_sec_lines = [line]
        else:
            current_sec_lines.append(line)

    if current_sec_lines:
        sections.append((current_sec_name, current_sec_lines))

    chunks = []

    # 2. Decompose high-level sections into fine-grained sub-chunks
    for sec_name, sec_lines in sections:
        sec_text = "\n".join(sec_lines)

        if sec_name == "Skills":
            for line in sec_lines:
                line_clean_hdr = line.rstrip(":").strip().lower()
                if line_clean_hdr in ["skills", "technical skills", "core competencies"]:
                    continue

                sub_parts = re.split(
                    r"\b(Languages|Tools|Frameworks|Soft\s+Skills|Core\s+Areas|Databases|Cloud|Platforms)[:\-]",
                    line,
                    flags=re.I
                )
                if len(sub_parts) > 1:
                    for i in range(1, len(sub_parts), 2):
                        cat_title = sub_parts[i].strip()
                        cat_content = sub_parts[i+1].strip() if i+1 < len(sub_parts) else ""
                        if cat_content:
                            full_line = f"{cat_title}: {cat_content}"
                            chunks.append({
                                "text": full_line,
                                "source": source,
                                "section": f"Skills ({cat_title.title()})",
                                "page": page_number
                            })
                            # Also add individual items if comma/bullet separated
                            items = [x.strip() for x in re.split(r"[,;•|]", cat_content) if x.strip()]
                            for it in items:
                                if len(it) >= 2 and it.lower() != full_line.lower():
                                    chunks.append({
                                        "text": f"{cat_title}: {it}",
                                        "source": source,
                                        "section": f"Skills ({cat_title.title()})",
                                        "page": page_number
                                    })
                else:
                    clean_l = re.sub(r"^\s*skills\s*:?\s*", "", line, flags=re.I).strip()
                    if clean_l and len(clean_l) > 2 and clean_l.lower() not in ["skills", "technical skills"]:
                        chunks.append({
                            "text": clean_l,
                            "source": source,
                            "section": "Skills",
                            "page": page_number
                        })
                        # Add fine-grained skill items if multiple exist
                        items = [x.strip() for x in re.split(r"[,;•|]", clean_l) if x.strip()]
                        if len(items) > 1:
                            for it in items:
                                if len(it) >= 2:
                                    chunks.append({
                                        "text": it,
                                        "source": source,
                                        "section": "Skills",
                                        "page": page_number
                                    })

        elif sec_name == "Projects":
            # Split by numbered projects (1., 2.), markdown headers, or Project Title lines
            proj_blocks = re.split(r"\n(?=\d+[\.\)]|\b[A-Z][A-Za-z0-9\s\-_]{2,40}:|\n)", sec_text)
            for p_block in proj_blocks:
                clean_p = p_block.strip()
                if clean_p.rstrip(":").strip().lower() in ["projects", "key projects", "academic projects"]:
                    continue
                if len(clean_p) >= 5:
                    final_p = re.sub(r"^projects:?\s*", "", clean_p, flags=re.I).strip()
                    if final_p:
                        chunks.append({
                            "text": final_p,
                            "source": source,
                            "section": "Projects",
                            "page": page_number
                        })

        elif sec_name == "Education":
            edu_blocks = sec_text.split("\n")
            for e_line in edu_blocks:
                clean_e = e_line.strip()
                if clean_e and len(clean_e) > 3 and clean_e.rstrip(":").strip().lower() not in ["education", "academics"]:
                    chunks.append({
                        "text": clean_e,
                        "source": source,
                        "section": "Education",
                        "page": page_number
                    })

        elif sec_name == "Experience":
            # Chunk each role or bullet point distinctly
            exp_blocks = [l.strip() for l in sec_lines if l.strip()]
            for e_line in exp_blocks:
                if e_line.rstrip(":").strip().lower() in ["experience", "work experience", "professional experience"]:
                    continue
                if len(e_line) >= 5:
                    chunks.append({
                        "text": e_line,
                        "source": source,
                        "section": "Experience",
                        "page": page_number
                    })

        elif sec_name == "Profile Summary":
            clean_s = re.sub(r"^profile\s+summary:?\s*", "", sec_text, flags=re.I).strip()
            if clean_s and len(clean_s) >= 5:
                chunks.append({
                    "text": clean_s,
                    "source": source,
                    "section": "Profile Summary",
                    "page": page_number
                })

        else:
            # Sub-split other blocks into concise lines or cohesive groups
            for sl in sec_lines:
                sl_clean = sl.strip()
                if not sl_clean or sl_clean.rstrip(":").strip().lower() in ["interests", "languages", "general"]:
                    continue
                if len(sl_clean) >= 3:
                    chunks.append({
                        "text": sl_clean,
                        "source": source,
                        "section": sec_name,
                        "page": page_number
                    })

    # Filter out duplicate, empty, or header-only chunks while preserving order
    final_chunks = []
    seen = set()
    for c in chunks:
        c_text = c["text"].strip()
        if len(c_text) < 2:
            continue
        key = (c_text.lower(), c["section"].lower(), c.get("page", 1))
        if key in seen:
            continue
        seen.add(key)
        final_chunks.append(c)

    return final_chunks


def chunk_text(text_or_pages, source: str = "resume") -> list[dict]:
    """
    Splits resume content into section-aware, concise evidence chunks.
    Accepts either a plain text string OR a list of page dicts [{"page_number": int, "text": str}].
    Attaches metadata: text, source, section, page.
    """
    if not text_or_pages:
        return []

    if isinstance(text_or_pages, list):
        # List of page dictionaries
        all_chunks = []
        for p in text_or_pages:
            p_num = p.get("page_number", 1)
            p_text = p.get("text", "")
            all_chunks.extend(_chunk_single_page(p_text, source=source, page_number=p_num))
        return all_chunks

    elif isinstance(text_or_pages, str):
        return _chunk_single_page(text_or_pages, source=source, page_number=1)

    return []