from rag.loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings, model
from rag.retriever import retrieve, match_requirements, categorize_matches
from rag.generator import generate_answer
from rag.requirements import extract_requirements


# 1. Load resume PDF
resume = load_document(
    "data/sample/resume.pdf"
)


# 2. Load job description PDF
job_description = load_document(
    "data/sample/job_description.pdf"
)


# 3. Extract job requirements dynamically
requirements = extract_requirements(
    job_description
)

print("\n--- Extracted Job Requirements ---")
for req in requirements:
    print(f"- {req}")


# 4. Split resume into section-aware chunks
chunks = chunk_text(
    resume,
    source="resume.pdf"
)


# 5. Create embeddings for resume chunks
embeddings = create_embeddings(
    chunks
)


# 6. Match job requirements against resume evidence
requirement_matches = match_requirements(
    requirements,
    embeddings,
    chunks,
    model,
    top_k=1
)


# 7. Categorize matches into 3 evidence states
analysis = categorize_matches(requirement_matches)

print("\n--- Match Analysis Dashboard ---")
print(f"Overall Match Coverage Score: {analysis['match_percentage']}%")
print(f"Total Requirements: {analysis['total_requirements']}")
print(f"Strong Evidence   : {analysis['strong_count']}")
print(f"Partial / Related : {analysis['partial_count']}")
print(f"Missing / Unverified: {analysis['missing_count']}")

print("\n--- Strong Evidence ---")
for item in analysis["strong_skills"]:
    print(f"✓ [STRONG EVIDENCE] {item['requirement']}")
    print(f"  Similarity Score: {item['score']:.4f}")
    print(f"  Resume Section  : {item['section']}")
    print(f"  Evidence Snippet: {item['evidence_text'][:120]}...\n")

print("\n--- Partial / Related Evidence ---")
for item in analysis["partial_skills"]:
    print(f"~ [PARTIAL / RELATED] {item['requirement']}")
    print(f"  Similarity Score: {item['score']:.4f}")
    print(f"  Resume Section  : {item['section']}")
    print(f"  Evidence Snippet: {item['evidence_text'][:120]}...\n")

print("\n--- Missing / Unverified Skills ---")
for item in analysis["missing_skills"]:
    print(f"✗ [MISSING / UNVERIFIED] {item['requirement']}")
    print(f"  Similarity Score: {item['score']:.4f}")
    print(f"  Evidence Status : {item['evidence_text']}\n")


# 8. Grounded LLM Query
question = "Summarize the candidate's core qualifications and list key gaps against the job description."

query_embedding = model.encode(question)
retrieved_chunks = retrieve(query_embedding, embeddings, chunks, top_k=2)

answer = generate_answer(retrieved_chunks, question)

print("\n--- Ollama Evidence-Grounded Answer ---")
print(answer)