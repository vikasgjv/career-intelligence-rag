# 🎯 Career Intelligence RAG

> An evidence-grounded resume analysis, requirement matching, and career compatibility engine powered by a two-tier evaluation architecture: deterministic structured eligibility verification and hybrid dense-sparse retrieval (Sentence Transformers + BM25Okapi) with local Ollama LLM advising.

---

## 🏗️ Architecture Overview

Unlike naive RAG systems that rely solely on dense vector search or ungrounded whole-resume prompting, **Career Intelligence RAG** utilizes a **two-tier evaluation engine**:

1. **Tier 1 — Deterministic Structured Eligibility:** Factual, rule-based criteria (CGPA cutoffs, academic percentages, graduation batch years, active backlogs, degrees, and branches) are evaluated deterministically using regular expressions and rule logic (`app/rag/structured_matcher.py`). Numeric comparisons ($8.63 \ge 6.0$) and strict absence checks are enforced without hallucination.
2. **Tier 2 — Hybrid Dense-Sparse RAG:** Technical skills, responsibilities, and domain competencies are evaluated via dense semantic vectors (`all-MiniLM-L6-v2`) and sparse keyword matching (`rank_bm25.BM25Okapi`) fused via **Reciprocal Rank Fusion (RRF, $k=60$)** (`app/rag/hybrid_retriever.py`).
3. **Exact Evidence Line Isolation:** Rather than returning entire multi-sentence paragraphs, `extract_exact_evidence_line()` isolates the specific supporting sentence from the candidate's resume for auditing.
4. **Local Evidence-Grounded Career Advisor:** Local **Ollama** (`llama3.2:3b`) provides interactive career advising constrained strictly to retrieved context.

```
                          User
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Upload Resume PDF             Upload Job Description PDF
            │                               │
            ▼                               ▼
   PDF Text Extraction             PDF Text Extraction
         (pypdf)                         (pypdf)
            │                               │
            ▼                               ▼
   Text Cleaning & Normalization   Dynamic Requirement Extraction
       (app/rag/cleaner.py)             (app/rag/requirements.py)
            │                               │
            ▼                               │
  Section-Aware Chunking                    │
  (Skills, Projects, Education)             │
       (app/rag/chunker.py)                 │
            │                               │
            ▼                               │
  Dense Embeddings Generation               │
  (all-MiniLM-L6-v2, 384-dim)              │
            │                               │
            └───────────────┬───────────────┘
                            ▼
              Two-Tier Evaluation Engine
                (app/rag/retriever.py)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  [Tier 1: Deterministic]          [Tier 2: Hybrid RAG]
  CGPA, Percentages, Backlogs,     Dense Cosine Similarity +
  Batch, Degree, Branch Aliases    BM25Okapi + Reciprocal Rank Fusion
  (structured_matcher.py)          (hybrid_retriever.py)
            │                               │
            └───────────────┬───────────────┘
                            ▼
               Exact Evidence Extraction &
               3-State Match Categorization
            (Strong, Partial, Missing Skills)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  Streamlit Web Dashboard         Context-Grounded Ollama LLM
   - 5 Executive Metrics           - llama3.2:3b local
   - Filterable Tabs & Badges      - Strict Grounding System Prompt
   - Line-Level Citations          (app/rag/generator.py)
```

---

## ✨ Key Features

- **📄 Native PDF Document Ingestion**: Ingests resume and job description PDFs page-by-page using `pypdf`, preserving page numbers.
- **🧹 Typography & Ligature Cleaning**: Standardizes bullets, separates embedded headers, and resolves typographic ligatures (`ﬁ` $\rightarrow$ `fi`) via NFKC Unicode normalization (`app/rag/cleaner.py`).
- **🏷️ Section-Aware Resume Chunking**: Partitions resumes into semantic sections (`Skills`, `Projects`, `Education`, `Experience`, `Profile Summary`) and decomposes individual skill items and projects into distinct, page-tagged evidence chunks (`app/rag/chunker.py`).
- **🎯 Dynamic Job Requirement Decomposition**: Filters corporate marketing, EEO boilerplate, and benefits sections; decomposes compound requirements (e.g. *"Programming in Java, C++, Python"*) into individual matchable items (`app/rag/requirements.py`).
- **🔢 Deterministic Eligibility Verification**: Accurately computes numeric academic cutoffs, degree aliases, branch mappings, and checks active backlogs (`app/rag/structured_matcher.py`).
- **⚡ Hybrid Dense-Sparse Retrieval**: Combines 384-dimensional dense vectors (`sentence-transformers/all-MiniLM-L6-v2`) with BM25 keyword matching via Reciprocal Rank Fusion ($k=60$), protecting technical terms like `C++` and `C#` from tokenizer stripping (`app/rag/hybrid_retriever.py`).
- **🔍 Exact Evidence Line Isolation**: Identifies and extracts the exact supporting sentence rather than whole chunks (`app/rag/retriever.py`).
- **🛡️ Privacy-Preserving Local LLM**: Employs local **Ollama** (`llama3.2:3b`) for conversational advice without external APIs or cloud data leakage (`app/rag/generator.py`).
- **🖥️ Streamlit Web Dashboard**: Features five executive metric cards, status badges, expandable evidence panels, and an AI career coach tab (`app/ui.py`).

---

## 🛠️ Technology Stack

| Component | Technology / Library | Role in Project |
| :--- | :--- | :--- |
| **Language Runtime** | Python 3.10+ (tested on Python 3.13) | Core application execution |
| **Web Dashboard** | Streamlit ($\ge 1.30.0$) | Interactive user interface |
| **PDF Extraction** | `pypdf` ($\ge 4.0.0$) | Page-by-page text extraction from PDF files and byte streams |
| **Text Normalization** | Python `unicodedata` & `re` | NFKC normalization, ligature conversion, and formatting |
| **Dense Embeddings** | `sentence-transformers` ($\ge 2.2.2$) | `all-MiniLM-L6-v2` generating 384-dim dense vectors (CPU) |
| **Vector Similarity** | NumPy ($\ge 1.24.0$) | In-memory vectorized cosine similarity |
| **Keyword Retrieval** | `rank-bm25` ($\ge 0.2.2$) | BM25Okapi sparse keyword scoring |
| **Rank Fusion** | Custom Implementation | Reciprocal Rank Fusion (RRF, $k=60$) blending dense & BM25 |
| **Local LLM** | Ollama Python SDK ($\ge 0.1.0$) | Connects to local `llama3.2:3b` model for grounded advising |
| **Testing** | Python `unittest` | Automated unit test suite (43 unit tests) |

---

## 🚀 Quick Start

### 1. Prerequisites
- macOS, Linux, or Windows (WSL2 recommended)
- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running locally with `llama3.2:3b`:
  ```bash
  ollama pull llama3.2:3b
  ```

### 2. Setup Virtual Environment
```bash
git clone https://github.com/your-username/career-intelligence-rag.git
cd career-intelligence-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
./venv/bin/python -m unittest discover tests
```
*Current test suite: 43 unit tests passing.*

### 4. Launch Streamlit Web UI
```bash
PYTHONPATH=app streamlit run app/ui.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Terminal Pipeline (Optional)
To test the pipeline directly in the terminal using the included sample documents:
```bash
PYTHONPATH=app python app/main.py
```

---

## 🧪 Project Structure

```
career-intelligence-rag/
├── app/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── cleaner.py             # PDF text normalization & ligature cleaning
│   │   ├── chunker.py             # Section-aware resume chunking engine
│   │   ├── config.py              # Similarity thresholds & scoring weights
│   │   ├── embeddings.py          # Sentence Transformer embedding pipeline
│   │   ├── generator.py           # Grounded Ollama LLM wrapper
│   │   ├── hybrid_retriever.py    # BM25 + Dense Reciprocal Rank Fusion (RRF)
│   │   ├── loader.py              # Unified PDF & text document loader
│   │   ├── requirements.py        # Dynamic JD requirement extraction & filtering
│   │   ├── retriever.py           # Two-tier matching, evidence isolation & scoring
│   │   └── structured_matcher.py  # Deterministic eligibility rules (CGPA, degree, etc.)
│   ├── main.py                    # Terminal pipeline runner
│   └── ui.py                      # Streamlit Web Dashboard application
├── data/
│   └── sample/                    # Synthetic sample PDF and TXT test files
│       ├── job_description.pdf
│       ├── job_description.txt
│       ├── resume.pdf
│       └── resume.txt
├── tests/                         # Automated unit test suite (43 tests)
│   ├── test_chunker.py
│   ├── test_cleaner.py
│   ├── test_hybrid_retriever.py
│   ├── test_loader.py
│   ├── test_requirements.py
│   ├── test_retriever.py
│   └── test_structured_matcher.py
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules for environments, caches & secrets
├── requirements.txt               # Application dependencies
└── README.md                      # Project documentation
```

---

## ⚖️ Known Limitations & Engineering Trade-offs

1. **Scanned / Image-Based PDFs**: Text extraction relies strictly on `pypdf`. Image-only scanned PDFs without embedded text layers require OCR (e.g. Tesseract), which is not currently implemented.
2. **Ephemeral In-Memory Storage**: Embeddings are computed dynamically in-memory per session using NumPy arrays. There is no persistent vector database (e.g., Chroma, FAISS).
3. **Reranker Interface**: The `HybridRetriever` class contains an extensible `reranker` callback parameter, but an automated cross-encoder model (e.g., MS MARCO) is not operational by default.
4. **Quantitative Evaluation Suite**: While the unit test suite thoroughly covers parsing, chunking, and retrieval logic (43 tests), formal RAG benchmark metrics (such as Precision@K, Recall@K, or Ragas faithfulness scores) are not yet implemented.
5. **Local LLM Dependency**: Tab 4 (Career Advisor) requires an active local Ollama daemon. If Ollama is offline, the core matching engine (Tabs 1–3) remains fully functional, but generating conversational advice requires starting the Ollama service.
