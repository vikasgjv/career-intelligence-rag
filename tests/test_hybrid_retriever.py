import unittest
from sentence_transformers import SentenceTransformer
from app.rag.hybrid_retriever import HybridRetriever, tokenize, reciprocal_rank_fusion

model = SentenceTransformer("all-MiniLM-L6-v2")


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {
                "text": "Skills: Python, SQL, Machine Learning, C++, FastAPI",
                "source": "resume.pdf",
                "section": "Skills",
                "page": 1
            },
            {
                "text": "Projects: Developed a high-performance REST API using FastAPI and Docker",
                "source": "resume.pdf",
                "section": "Projects",
                "page": 2
            },
            {
                "text": "Experience: Senior Backend Developer using Java, Kubernetes, and AWS",
                "source": "resume.pdf",
                "section": "Experience",
                "page": 2
            }
        ]
        texts = [c["text"] for c in self.chunks]
        self.embeddings = model.encode(texts)
        self.retriever = HybridRetriever(self.chunks, embeddings=self.embeddings, model=model)

    def test_tokenize_preserves_special_tech_terms(self):
        tokens = tokenize("Programming in C++, C#, and Node.js")
        self.assertIn("c++", tokens)
        self.assertIn("c#", tokens)
        self.assertIn("node.js", tokens)

    def test_bm25_exact_keyword_retrieval(self):
        results = self.retriever.retrieve_bm25("FastAPI", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        # Chunks containing FastAPI should be retrieved
        retrieved_texts = [r["chunk"]["text"] for r in results]
        self.assertTrue(any("FastAPI" in t for t in retrieved_texts))

    def test_dense_semantic_retrieval(self):
        results = self.retriever.retrieve_dense("building web service endpoints", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("score", results[0])
        self.assertGreater(results[0]["score"], 0.0)

    def test_reciprocal_rank_fusion_scoring_and_deduplication(self):
        dense_results = [
            {"index": 0, "chunk": self.chunks[0], "score": 0.8},
            {"index": 1, "chunk": self.chunks[1], "score": 0.5}
        ]
        bm25_results = [
            {"index": 1, "chunk": self.chunks[1], "score": 2.5},
            {"index": 2, "chunk": self.chunks[2], "score": 1.2}
        ]
        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

        # Unique indices only (deduplication)
        indices = [item["index"] for item in fused]
        self.assertEqual(len(indices), len(set(indices)))

        # Index 1 was retrieved by BOTH systems, so its RRF score should be boosted
        item_1 = next(item for item in fused if item["index"] == 1)
        self.assertEqual(item_1["retrieval_method"], "hybrid_rrf")
        self.assertGreater(item_1["rrf_score"], 0.0)

        # Verify metadata retention
        self.assertIn("section", item_1)
        self.assertIn("page", item_1)
        self.assertIn("source", item_1)

    def test_hybrid_search_end_to_end(self):
        results = self.retriever.retrieve_hybrid("FastAPI REST API", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("FastAPI", top_match["text"])
        self.assertIn(top_match["retrieval_method"], ["hybrid_rrf", "keyword_bm25", "semantic_dense"])

    def test_modular_reranker_hook(self):
        # Define a mock reranker callback that reverses the results
        def mock_reranker(candidates, query):
            return list(reversed(candidates))

        retriever_with_reranker = HybridRetriever(
            self.chunks,
            embeddings=self.embeddings,
            model=model,
            reranker=mock_reranker
        )
        results = retriever_with_reranker.retrieve_hybrid("Python", top_k=3)
        self.assertGreaterEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
