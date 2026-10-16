from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from .cleaner import clean_text


def load_pdf_pages(file_input) -> list[dict]:
    """
    Extract text page-by-page from a PDF file path or file-like object.
    Returns a list of dicts: [{"page_number": int, "text": str}]
    Raises ValueError if the PDF is corrupt, encrypted, or unreadable.
    """
    try:
        if isinstance(file_input, (str, Path)):
            reader = PdfReader(str(file_input))
        elif hasattr(file_input, "read"):
            # File-like object (e.g. BytesIO or Streamlit UploadedFile)
            reader = PdfReader(file_input)
        else:
            raise ValueError("Invalid file input: must be a file path or file-like object.")

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("PDF is password-protected and cannot be read.")

        pages_data = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                text = ""
            clean_page_text = clean_text(text)
            if clean_page_text:
                pages_data.append({
                    "page_number": idx + 1,
                    "text": clean_page_text
                })

        return pages_data

    except PdfReadError as e:
        raise ValueError(f"Corrupt or invalid PDF file: {e}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to read PDF document: {e}")


def load_pdf(file_input) -> str:
    """Extract and combine text from all pages of a PDF file."""
    pages = load_pdf_pages(file_input)
    if not pages:
        return ""
    return "\n\n".join(p["text"] for p in pages)


def load_text(file_path: str) -> str:
    """Load text content from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def load_document(file_path: str) -> str:
    """
    Unified document loader for file paths.
    Detects file extension and routes to appropriate handler (.pdf or .txt).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        raw_text = load_pdf(str(path))
    elif ext == ".txt":
        raw_text = load_text(str(path))
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: .pdf, .txt"
        )

    return clean_text(raw_text)


def load_document_pages(file_input, filename: str = "") -> list[dict]:
    """
    Unified document loader returning page-aware chunks.
    Works with file paths or file-like objects.
    """
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return load_pdf_pages(str(path))
        elif ext == ".txt":
            raw_text = clean_text(load_text(str(path)))
            return [{"page_number": 1, "text": raw_text}] if raw_text else []
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported formats: .pdf, .txt")

    # File-like object
    ext = Path(filename).suffix.lower() if filename else ".pdf"
    if ext == ".pdf":
        return load_pdf_pages(file_input)
    else:
        content = file_input.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        cleaned = clean_text(text)
        return [{"page_number": 1, "text": cleaned}] if cleaned else []