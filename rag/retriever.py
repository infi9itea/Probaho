import torch
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


class Retriever:
    def __init__(self, vectorstore_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Snippet uses cuda:1 for reranker if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            reranker_device = "cuda:1"
        elif torch.cuda.is_available():
            reranker_device = "cuda:0"
        else:
            reranker_device = "cpu"

        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=reranker_device
        )

    def retrieve(self, query: str, top_k: int = 25, return_k: int = 8) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
        Implemented with Dynamic Priority logic from v2 snippet.
        """
        retrieved_docs = self.vectorstore.similarity_search(query, k=top_k)
        if not retrieved_docs:
            return []

        dynamic_priority = [d for d in retrieved_docs if "ewubd.edu" in d.metadata.get("source", "")]
        static_others = [d for d in retrieved_docs if "ewubd.edu" not in d.metadata.get("source", "")]

        if dynamic_priority:
            # As per snippet: rerank top return_k dynamic priority docs
            docs_to_rerank = dynamic_priority[:return_k]
        else:
            # As per snippet: rerank top return_k other docs
            docs_to_rerank = static_others[:return_k]

        if not docs_to_rerank:
            return []

        pairs = [(query, d.page_content) for d in docs_to_rerank]
        scores = self.reranker.predict(pairs, batch_size=32).tolist()

        ranked = sorted(
            zip(docs_to_rerank, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "score": float(score)
            }
            for doc, score in ranked[:return_k]
        ]
        return results
