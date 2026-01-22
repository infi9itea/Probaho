import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

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

    def retrieve(self, query: str, top_k=6):
        # Use simple similarity search (no score filtering)
        docs = self.vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return []

        texts = [d.page_content for d in docs]
        pairs = [(query, t) for t in texts]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(texts, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [t for t, _ in ranked[:3]]