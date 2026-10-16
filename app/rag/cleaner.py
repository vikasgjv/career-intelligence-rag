import re
import unicodedata

KNOWN_HEADERS = {
    "skills", "technical skills", "projects", "key projects", "education",
    "academic background", "experience", "work experience", "professional experience",
    "summary", "profile summary", "qualifications", "requirements", "responsibilities",
    "eligibility criteria", "about us", "benefits", "certifications", "languages",
    "interests", "profile"
}

SUBHEADER_PREFIXES = [
    "languages", "tools", "frameworks", "soft skills", "core areas",
    "databases", "cloud", "platforms"
]


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDF or TXT files.
    - Normalizes unicode characters and ligatures.
    - Removes common PDF page header/footer artifacts.
    - Standardizes bullet symbols.
    - Reconstructs broken line wraps while preserving section boundaries.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Standardize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip page header/footer artifacts
    text = re.sub(r"(?i)page\s+\d+(\s+of\s+\d+)?", "", text)

    # 4. Standardize common PDF bullet symbols
    text = re.sub(r"[•·▪▸▶\t]", "\n- ", text)

    # 5. Separate embedded major headers with newlines
    for hdr in KNOWN_HEADERS:
        text = re.sub(rf"(?<=[^\n])\s*\b({re.escape(hdr)})\b(?!\:)", r"\n\1", text, flags=re.I)

    for sub in SUBHEADER_PREFIXES:
        text = re.sub(rf"(?<=[^\n])\s*\b({re.escape(sub)})[:\-]", r"\n\1:", text, flags=re.I)


    # 6. Clean extra horizontal spaces line by line
    raw_lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]
    raw_lines = [line for line in raw_lines if line]

    if not raw_lines:
        return ""

    processed_lines = []
    current_line = ""

    for line in raw_lines:
        line_clean = line.rstrip(":").strip().lower()

        is_bullet = line.startswith("- ") or line.startswith("* ") or bool(re.match(r"^\d+[\.\)]\s+", line))
        is_header = line_clean in KNOWN_HEADERS or any(line_clean.startswith(sub) for sub in SUBHEADER_PREFIXES) or line.endswith(":")

        if not current_line:
            current_line = line
        elif is_bullet or is_header or current_line.endswith((".", ":", ";", "!", "?")):
            processed_lines.append(current_line)
            current_line = line
        else:
            current_line = f"{current_line} {line}"

    if current_line:
        processed_lines.append(current_line)

    final_lines = []
    for line in processed_lines:
        header_key = line.rstrip(":").strip().lower()
        if header_key in KNOWN_HEADERS:
            if not line.endswith(":"):
                line = f"{line.rstrip(':')}:"
            if final_lines and final_lines[-1] != "":
                final_lines.append("")

        final_lines.append(line)

    cleaned_text = "\n".join(final_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
