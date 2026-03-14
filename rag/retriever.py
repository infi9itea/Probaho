import torch
from langchain_community.vectorstores import FAISS
# ← revert this line
from langchain_community.embeddings import HuggingFaceEmbeddings      # ← updated
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


class Retriever:
    def __init__(self, vectorstore_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def retrieve(self, query: str, top_k: int = 30, return_k: int = 6) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return []

        texts = [d.page_content for d in docs]
        sources = [d.metadata.get("source", "") for d in docs]

        # Cross-encoder reranking
        pairs = [(query, t) for t in texts]
        scores = self.reranker.predict(pairs, batch_size=32).tolist()

        # Sort by relevance (higher score = more relevant)
        ranked = sorted(
            zip(texts, sources, scores),
            key=lambda x: x[2],
            reverse=True
        )

        results = [
            {"text": t, "source": s, "score": float(sc)}
            for t, s, sc in ranked[:return_k]
        ]
        return results
