import unittest
from sentence_transformers import SentenceTransformer
from app.rag.retriever import retrieve, match_requirements, categorize_matches, extract_exact_evidence_line

model = SentenceTransformer("all-MiniLM-L6-v2")


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"text": "Skills:\nPython, SQL, Machine Learning", "source": "resume", "section": "Skills", "page": 1},
            {"text": "Projects:\nBuilt RAG system using Ollama and Python", "source": "resume", "section": "Projects", "page": 2}
        ]
        texts = [c["text"] for c in self.chunks]
        self.embeddings = model.encode(texts)

    def test_retrieve_cosine_similarity(self):
        query_emb = model.encode("Python programming")
        results = retrieve(query_emb, self.embeddings, self.chunks, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("score", results[0])
        self.assertGreater(results[0]["score"], 0.0)
        self.assertEqual(results[0]["retrieval_method"], "semantic_dense")

    def test_page_and_method_metadata(self):
        query_emb = model.encode("RAG system with Ollama")
        results = retrieve(query_emb, self.embeddings, self.chunks, top_k=1)
        self.assertEqual(results[0]["page"], 2)
        self.assertEqual(results[0]["section"], "Projects")

    def test_exact_evidence_line_isolation(self):
        multiline_chunk = "Career Intelligence RAG\nDeveloped machine learning pipelines using Python.\nIntegrated FastAPI and Docker."
        isolated = extract_exact_evidence_line(multiline_chunk, "Python")
        self.assertIn("Python", isolated)
        self.assertEqual(isolated, "Developed machine learning pipelines using Python.")

    def test_categorize_matches_3_states(self):
        reqs = ["Python", "Docker"]
        matches = match_requirements(reqs, self.embeddings, self.chunks, model)
        summary = categorize_matches(matches)

        self.assertIn("strong_skills", summary)
        self.assertIn("partial_skills", summary)
        self.assertIn("missing_skills", summary)
        self.assertEqual(summary["total_requirements"], 2)

    def test_prevent_irrelevant_evidence_for_missing_skills(self):
        reqs = ["Docker"]
        matches = match_requirements(reqs, self.embeddings, self.chunks, model)
        evidence_text = matches[0]["evidence"][0]["text"]
        self.assertEqual(matches[0]["status"], "MISSING / UNVERIFIED")
        self.assertEqual(evidence_text, "No strong evidence found in the resume.")


if __name__ == "__main__":
    unittest.main()
