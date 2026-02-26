import numpy as np
from tqdm import tqdm

from infra.chat_factory import EmbeddingModel
from infra.models import Chunk, RetrievedChunk, FocusSpec

BATCH_SIZE = 64


class ChunkIndex:
    """FAISS-backed vector index for chunk retrieval."""

    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.index = None  # faiss.IndexFlatIP

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build FAISS index."""
        import faiss

        self.chunks = chunks
        texts = [c.text for c in chunks]

        all_embeddings: list[list[float]] = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
            batch = texts[i : i + BATCH_SIZE]
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)

        # Inner product on L2-normalized vectors = cosine similarity
        matrix = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)

    def retrieve(
        self,
        focus_spec: FocusSpec,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks matching focus queries. Merges and deduplicates."""
        import faiss

        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        all_queries = (
            focus_spec.retrieval_queries
            + [focus_spec.primary_focus]
            + focus_spec.keywords
        )

        best_scores: dict[str, float] = {}

        query_embeddings = np.array(self.embedder.embed(all_queries), dtype=np.float32)
        faiss.normalize_L2(query_embeddings)

        scores, indices = self.index.search(query_embeddings, top_k)

        for q_idx in range(len(all_queries)):
            for rank in range(top_k):
                idx = int(indices[q_idx][rank])
                if idx < 0:
                    continue
                score = float(scores[q_idx][rank])
                chunk = self.chunks[idx]
                cid = chunk.chunk_id
                if cid not in best_scores or score > best_scores[cid]:
                    best_scores[cid] = score

        # Build lookup for fast access
        chunk_by_id = {c.chunk_id: c for c in self.chunks}

        results = []
        for cid, score in sorted(best_scores.items(), key=lambda x: -x[1]):
            chunk = chunk_by_id[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                )
            )
            if len(results) >= top_k:
                break

        return results
