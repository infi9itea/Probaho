import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(
            model_name,
            device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )

    def rerank(self, query: str, docs: List[Any], top_k: int = 3) -> List[Any]:
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs).tolist()

        scored_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [{"text": doc.page_content, "source": doc.metadata.get("source", ""), "score": float(score)} for doc, score in scored_docs[:top_k]]


class Retriever:
    def __init__(self, vectorstore_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 25, return_k: int = 8) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
        Using Dynamic Priority logic.
        """
        retrieved_docs = self.vectorstore.similarity_search(query, k=top_k)
        if not retrieved_docs:
            return []

        dynamic_priority = [d for d in retrieved_docs if "ewubd.edu" in d.metadata.get("source", "")]
        static_others = [d for d in retrieved_docs if "ewubd.edu" not in d.metadata.get("source", "")]

        if dynamic_priority:
            # If we have official site hits, we prioritize them for reranking
            reranked = self.reranker.rerank(query, dynamic_priority, top_k=return_k)
        else:
            # Otherwise rerank from other sources (JSON files)
            reranked = self.reranker.rerank(query, static_others, top_k=return_k)

        return reranked
