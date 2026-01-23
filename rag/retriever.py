import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Retriever:
    def __init__(self, vectorstore_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def retrieve(self, query: str, top_k=25, return_k=3) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: [{ "text": ..., "source": ..., "score": ... }, ...]
        - top_k: number of candidates to pull from FAISS
        - return_k: how many final passages to return (after reranking)
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return []

        texts = [d.page_content for d in docs]
        sources = [d.metadata.get("source", "") if hasattr(d, "metadata") else "" for d in docs]
        pairs = [(query, t) for t in texts]

        scores = self.reranker.predict(pairs).tolist()

        ranked = sorted(
            zip(texts, sources, scores),
            key=lambda x: x[2],
            reverse=True
        )

        results = [{"text": t, "source": s, "score": float(sc)} for t, s, sc in ranked[:return_k]]
        return results