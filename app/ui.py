import streamlit as st
import numpy as np

from rag.loader import load_pdf_pages
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings, model
from rag.retriever import retrieve, match_requirements, categorize_matches
from rag.hybrid_retriever import HybridRetriever
from rag.generator import generate_answer
from rag.requirements import extract_requirements
from rag.config import STRONG_EVIDENCE_THRESHOLD, PARTIAL_EVIDENCE_THRESHOLD


st.set_page_config(
    page_title="Career Intelligence RAG",
    page_icon="💼",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .strong-badge {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .partial-badge {
        background-color: #ffc107;
        color: #333;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .missing-badge {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .category-badge {
        background-color: #0288d1;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    /* Hide ONLY the default 200MB limit text inside Streamlit dropzone */
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Career Intelligence RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-Grounded Resume Analysis & Job Compatibility Engine</div>', unsafe_allow_html=True)

# Sidebar - Document Upload
st.sidebar.header("📁 Document Upload")

resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
st.sidebar.caption("PDF only — Maximum 10 MB per file")

jd_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type=["pdf"])
st.sidebar.caption("PDF only — Maximum 10 MB per file")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

if resume_file and resume_file.size > MAX_FILE_SIZE:
    st.sidebar.error("Resume file exceeds the 10 MB limit.")

if jd_file and jd_file.size > MAX_FILE_SIZE:
    st.sidebar.error("Job Description file exceeds the 10 MB limit.")

if st.sidebar.button("🚀 Analyze Compatibility", type="primary", use_container_width=True):
    if not resume_file or not jd_file:
        st.error("Please upload both Resume (PDF) and Job Description (PDF) files.")
    elif resume_file.size > MAX_FILE_SIZE or jd_file.size > MAX_FILE_SIZE:
        st.error("One or more files exceed the maximum 10 MB file size limit.")
    else:
        try:
            with st.spinner("Processing PDF Documents..."):
                resume_pages = load_pdf_pages(resume_file)
                jd_pages = load_pdf_pages(jd_file)

            if not resume_pages or not any(p["text"] for p in resume_pages):
                st.error("Uploaded Resume PDF does not contain readable text. Please check the document.")
            elif not jd_pages or not any(p["text"] for p in jd_pages):
                st.error("Uploaded Job Description PDF does not contain readable text. Please check the document.")
            else:
                st.session_state["analyzed"] = True
                st.session_state["resume_pages"] = resume_pages
                st.session_state["resume_text"] = "\n\n".join(p["text"] for p in resume_pages)
                st.session_state["jd_text"] = "\n\n".join(p["text"] for p in jd_pages)
        except ValueError as e:
            st.error(f"Error reading PDF document: {e}")
        except Exception as e:
            st.error(f"Unexpected error processing PDF documents: {e}")


if st.session_state.get("analyzed", False):
    resume_pages = st.session_state["resume_pages"]
    resume_text = st.session_state["resume_text"]
    jd_text = st.session_state["jd_text"]

    with st.spinner("Analyzing Evidence and Matching Requirements..."):
        requirements = extract_requirements(jd_text)
        chunks = chunk_text(resume_pages, source="Uploaded Resume")
        embeddings = create_embeddings(chunks)

        requirement_matches = match_requirements(
            requirements,
            embeddings,
            chunks,
            model,
            top_k=1,
            full_resume_text=resume_text
        )

        analysis = categorize_matches(requirement_matches)

    # Executive Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Overall Match Coverage", f"{analysis['match_percentage']}%", help="Evidence Coverage Score weighted by match confidence")
    with col2:
        st.metric("Total Requirements", analysis["total_requirements"])
    with col3:
        st.metric("Strong Evidence (≥0.40)", analysis["strong_count"])
    with col4:
        st.metric("Partial / Related", analysis["partial_count"])
    with col5:
        st.metric("Missing / Unverified", analysis["missing_count"])

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "✅ Strong Evidence",
        "⚠️ Partial / Related",
        "❌ Missing / Unverified Skills",
        "🤖 Ollama Career Advisor"
    ])

    with tab1:
        st.subheader("Strong Evidence Requirements")
        if not analysis["strong_skills"]:
            st.info("No requirements met the strong evidence criteria.")
        for item in analysis["strong_skills"]:
            cat_label = item.get("category", "technical_skill").replace("_", " ").title()
            method_label = item.get("match_method", "semantic_dense").replace("_", " ").title()
            with st.expander(f"✓ [{cat_label}] {item['requirement']} (Score: {item['score']:.4f})"):
                st.markdown(f"**Status**: <span class='strong-badge'>STRONG EVIDENCE</span> &nbsp;|&nbsp; **Category**: <span class='category-badge'>{cat_label.upper()}</span> &nbsp;|&nbsp; **Method**: `{method_label}`", unsafe_allow_html=True)
                if item.get("candidate_value") is not None and item.get("required_value") is not None:
                    st.markdown(f"**Required Value**: `{item['required_value']}` &nbsp;|&nbsp; **Candidate Value**: `{item['candidate_value']}`")
                st.markdown(f"**JD Section**: `{item.get('source_section', 'N/A')}` &nbsp;|&nbsp; **Resume Section**: `{item['section']}` &nbsp;|&nbsp; **Page**: `{item.get('page', 1)}`")
                st.markdown("**Exact Resume Evidence**:")
                st.code(item["evidence_text"], language="text")

    with tab2:
        st.subheader("Partial / Related Evidence (Similarity 0.30 - 0.39)")
        if not analysis["partial_skills"]:
            st.info("No partial/related requirements in this category.")
        for item in analysis["partial_skills"]:
            cat_label = item.get("category", "technical_skill").replace("_", " ").title()
            method_label = item.get("match_method", "semantic_dense").replace("_", " ").title()
            with st.expander(f"~ [{cat_label}] {item['requirement']} (Score: {item['score']:.4f})"):
                st.markdown(f"**Status**: <span class='partial-badge'>PARTIAL / RELATED</span> &nbsp;|&nbsp; **Category**: <span class='category-badge'>{cat_label.upper()}</span> &nbsp;|&nbsp; **Method**: `{method_label}`", unsafe_allow_html=True)
                st.markdown(f"**JD Section**: `{item.get('source_section', 'N/A')}` &nbsp;|&nbsp; **Resume Section**: `{item['section']}` &nbsp;|&nbsp; **Page**: `{item.get('page', 1)}`")
                st.markdown("**Exact Resume Evidence**:")
                st.code(item["evidence_text"], language="text")

    with tab3:
        st.subheader("Requirements Needing Evidence (Similarity < 0.30)")
        if not analysis["missing_skills"]:
            st.success("All requirements are supported by strong or partial evidence!")
        for item in analysis["missing_skills"]:
            cat_label = item.get("category", "technical_skill").replace("_", " ").title()
            method_label = item.get("match_method", "none").replace("_", " ").title()
            st.markdown(f"- <span class='missing-badge'>MISSING / UNVERIFIED</span> <span class='category-badge'>{cat_label.upper()}</span> **{item['requirement']}** *(Method: `{method_label}`)*", unsafe_allow_html=True)
            st.markdown(f"  *Evidence*: `{item['evidence_text']}`")

    with tab4:
        st.subheader("Evidence-Grounded AI Career Advice (Ollama llama3.2:3b)")
        custom_question = st.text_input(
            "Ask a targeted question about candidate fit:",
            value="What are the candidate's strongest qualifications and top 2 actionable recommendations to improve match?"
        )
        if st.button("Generate Advice", type="secondary"):
            with st.spinner("Generating grounded answer via Ollama..."):
                hr = HybridRetriever(chunks, embeddings=embeddings, model=model)
                retrieved_chunks = hr.retrieve_hybrid(custom_question, top_k=3)
                answer = generate_answer(retrieved_chunks, custom_question)
                st.markdown("### AI Career Analysis")
                st.write(answer)

