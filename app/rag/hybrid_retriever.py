import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Tokenizes text while preserving technical keywords and symbols (e.g., C++, C#, .NET).
    """
    if not text:
        return []
    text_clean = text.lower()
    # Normalize symbols with unique tokens to preserve them across word boundaries
    text_clean = text_clean.replace("c++", " __cpp__ ").replace("c#", " __csharp__ ")
    raw_tokens = re.findall(r"\b[a-z0-9_\-\.]{2,}\b|\b[a-z]\b", text_clean)

    tokens = []
    for t in raw_tokens:
        if t == "__cpp__":
            tokens.append("c++")
        elif t == "__csharp__":
            tokens.append("c#")
        else:
            tokens.append(t)
    return tokens


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0
) -> list[dict]:
    """
    Combines dense semantic rankings and BM25 keyword rankings using
    Reciprocal Rank Fusion (RRF).
    Formula: RRF(d) = sum(weight_m / (k + rank_m(d))) for each retrieval system m.
    Deduplicates results and preserves metadata (section, page, source).
    """
    fusion_map = {}

    # 1. Process dense rankings
    for rank, item in enumerate(dense_results, start=1):
        idx = item["index"]
        if idx not in fusion_map:
            fusion_map[idx] = {
                "chunk": item["chunk"],
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_score": item["score"],
                "bm25_score": 0.0,
                "methods": ["semantic_dense"]
            }
        else:
            fusion_map[idx]["dense_rank"] = rank
            fusion_map[idx]["dense_score"] = item["score"]
            if "semantic_dense" not in fusion_map[idx]["methods"]:
                fusion_map[idx]["methods"].append("semantic_dense")

    # 2. Process BM25 rankings
    for rank, item in enumerate(bm25_results, start=1):
        idx = item["index"]
        if idx not in fusion_map:
            fusion_map[idx] = {
                "chunk": item["chunk"],
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_score": 0.0,
                "bm25_score": item["score"],
                "methods": ["keyword_bm25"]
            }
        else:
            fusion_map[idx]["bm25_rank"] = rank
            fusion_map[idx]["bm25_score"] = item["score"]
            if "keyword_bm25" not in fusion_map[idx]["methods"]:
                fusion_map[idx]["methods"].append("keyword_bm25")

    # 3. Compute RRF scores
    fused_items = []
    for idx, data in fusion_map.items():
        rrf_score = 0.0
        if data["dense_rank"] is not None:
            rrf_score += dense_weight / (k + data["dense_rank"])
        if data["bm25_rank"] is not None:
            rrf_score += bm25_weight / (k + data["bm25_rank"])

        methods = data["methods"]
        if len(methods) > 1:
            method_label = "hybrid_rrf"
        else:
            method_label = methods[0]

        chunk = data["chunk"]
        fused_items.append({
            "index": idx,
            "text": chunk["text"],
            "source": chunk.get("source", "resume"),
            "section": chunk.get("section", "General"),
            "page": chunk.get("page", 1),
            "rrf_score": rrf_score,
            "dense_score": data["dense_score"],
            "bm25_score": data["bm25_score"],
            "score": data["dense_score"],  # Preserve calibrated cosine score as baseline score
            "retrieval_method": method_label
        })

    # 4. Sort descending by RRF score, with dense score as tie-breaker
    fused_items.sort(key=lambda x: (x["rrf_score"], x["dense_score"]), reverse=True)
    return fused_items


class HybridRetriever:
    """
    Modular Hybrid Search Engine combining Sentence Transformer dense embeddings
    with BM25 keyword retrieval fused via Reciprocal Rank Fusion.
    Allows pluggable reranking stages.
    """

    def __init__(self, chunks: list[dict], embeddings=None, model=None, k: int = 60, reranker=None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.k = k
        self.reranker = reranker

        # Initialize BM25 index on chunk corpus
        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        if self.tokenized_corpus and any(len(doc) > 0 for doc in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve_dense(self, query: str, query_embedding=None, top_k: int = 5) -> list[dict]:
        """Performs dense vector retrieval using cosine similarity."""
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        if query_embedding is None:
            if self.model is None:
                raise ValueError("Embedding model is required to compute query embedding.")
            query_embedding = self.model.encode(query)

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        embeddings_norm = self.embeddings / (np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        ) + 1e-9)

        similarities = np.dot(embeddings_norm, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for index in top_indices:
            results.append({
                "index": int(index),
                "chunk": self.chunks[index],
                "score": float(similarities[index]),
                "method": "semantic_dense"
            })
        return results

    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs BM25 keyword retrieval."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for index in top_indices:
            score = float(doc_scores[index])
            if score > 0.0:
                results.append({
                    "index": int(index),
                    "chunk": self.chunks[index],
                    "score": score,
                    "method": "keyword_bm25"
                })
        return results

    def retrieve_hybrid(self, query: str, query_embedding=None, top_k: int = 2) -> list[dict]:
        """
        Executes dense retrieval and BM25 retrieval, fusions rankings via RRF,
        applies reranker if configured, and returns top_k evidence chunks.
        """
        if not self.chunks:
            return []

        dense_top = self.retrieve_dense(query, query_embedding=query_embedding, top_k=max(top_k * 3, 5))
        bm25_top = self.retrieve_bm25(query, top_k=max(top_k * 3, 5))

        fused = reciprocal_rank_fusion(dense_top, bm25_top, k=self.k)

        # Pluggable reranker stage
        if self.reranker and callable(self.reranker):
            fused = self.reranker(fused, query)

        return fused[:top_k]
