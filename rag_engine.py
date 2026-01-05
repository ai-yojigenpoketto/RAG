"""
Advanced RAG engine that wires together dense HNSW search, BM25 keyword search,
score fusion, cross-encoder re-ranking, and OpenAI generation.

Designed for clarity and interview-ready talking points rather than obscuring
the retrieval logic behind higher-level abstractions.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


class AdvancedRAGEngine:
    """
    An end-to-end RAG engine that exposes ingestion, hybrid retrieval,
    re-ranking, and generation steps transparently.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 64,
        fusion_strategy: str = "weighted",  # "weighted" or "rrf"
        normalization: str = "minmax",  # "minmax" or "zscore"
        rrf_k: int = 60,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.fusion_strategy = fusion_strategy
        self.normalization = normalization
        self.rrf_k = rrf_k

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=30
        )
        # HuggingFace embeddings stay local to satisfy data residency concerns.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        # Cross-encoder scores query-document relevance better than raw cosine
        # similarity, providing a learn-to-rank style signal.
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm: Optional[ChatOpenAI] = None

        self.documents: List[Document] = []
        self.vector_store: Optional[FAISS] = None
        self.bm25_index: Optional[BM25Okapi] = None

    def ingest_texts(self, texts: List[str]) -> None:
        """
        Accept raw text strings, chunk them, and store as Documents with metadata.
        """
        docs: List[Document] = []
        doc_id = 0
        for source_id, text in enumerate(texts):
            if not text or not text.strip():
                continue
            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": doc_id,
                            "source_id": source_id,
                            "chunk_id": chunk_idx,
                        },
                    )
                )
                doc_id += 1

        self.documents = docs

    def build_indexes(self) -> None:
        """
        Build both dense (FAISS HNSW) and sparse (BM25) indexes.
        """
        if not self.documents:
            raise ValueError("No documents ingested. Call ingest_texts() first.")

        # Dense index: HNSW for sub-linear search and solid recall at low latency.
        embeddings = self.embeddings.embed_documents(
            [doc.page_content for doc in self.documents]
        )
        embedding_matrix = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embedding_matrix)

        dim = embedding_matrix.shape[1]
        hnsw_index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efConstruction = self.hnsw_ef_construction
        hnsw_index.hnsw.efSearch = self.hnsw_ef_search

        # Use IDMap so FAISS keeps our doc_id association.
        index = faiss.IndexIDMap(hnsw_index)
        ids = np.array([doc.metadata["doc_id"] for doc in self.documents]).astype(
            np.int64
        )
        index.add_with_ids(embedding_matrix, ids)

        index_to_docstore_id = {int(i): str(int(i)) for i in ids}
        docstore = InMemoryDocstore(
            {doc_id: doc for doc_id, doc in zip(index_to_docstore_id.values(), self.documents)}
        )
        self.vector_store = FAISS(
            embedding_function=self.embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True,
        )

        # Sparse index via BM25 for exact keyword hits and lexical diversity.
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def is_ready(self) -> bool:
        return bool(self.vector_store and self.bm25_index)

    def retrieve(
        self, query: str, top_k: int = 4, alpha: float = 0.5
    ) -> Tuple[List[Document], Dict[str, List[Dict[str, float]]]]:
        if not self.is_ready():
            raise ValueError("Indexes not built. Ingest text and call build_indexes().")

        # Vector search returns similarity scores from FAISS (higher is better).
        vector_hits = self.vector_store.similarity_search_with_score(
            query, k=max(1, top_k * 2)
        )
        vector_scores = {
            doc.metadata["doc_id"]: float(score) for doc, score in vector_hits
        }
        vector_docs = {
            doc.metadata["doc_id"]: doc for doc, _ in vector_hits
        }

        # BM25 search over tokenized corpus.
        query_tokens = self._tokenize(query)
        bm25_scores_raw = self.bm25_index.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores_raw)[::-1][: max(1, top_k * 2)]
        bm25_scores: Dict[int, float] = {}
        bm25_docs: Dict[int, Document] = {}
        for idx in bm25_top_indices:
            doc = self.documents[int(idx)]
            bm25_scores[int(idx)] = float(bm25_scores_raw[idx])
            bm25_docs[int(idx)] = doc

        # Normalize scores for fair fusion.
        norm_vec = self._normalize_scores(vector_scores, method=self.normalization)
        norm_bm25 = self._normalize_scores(bm25_scores, method=self.normalization)

        fused_scores: Dict[int, float] = {}
        if self.fusion_strategy == "rrf":
            fused_scores = self._rrf_fuse(norm_vec, norm_bm25, k=self.rrf_k)
        else:
            candidate_ids = list(set(norm_vec.keys()) | set(norm_bm25.keys()))
            vec_arr = np.array([norm_vec.get(i, 0.0) for i in candidate_ids], dtype=np.float32)
            bm25_arr = np.array([norm_bm25.get(i, 0.0) for i in candidate_ids], dtype=np.float32)
            fused_arr = alpha * vec_arr + (1 - alpha) * bm25_arr
            fused_scores = {doc_id: float(score) for doc_id, score in zip(candidate_ids, fused_arr.tolist())}

        # Take strongest fused candidates, then re-rank with cross-encoder.
        sorted_candidates = sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )
        rerank_pool = [
            (doc_id, vector_docs.get(doc_id) or bm25_docs.get(doc_id))
            for doc_id, _ in sorted_candidates[: max(1, top_k * 2)]
        ]
        pairs = [(query, doc.page_content) for _, doc in rerank_pool]
        rerank_scores = self.cross_encoder.predict(pairs).tolist()

        reranked = sorted(
            [
                {"doc": doc, "doc_id": doc_id, "score": float(score), "fused": fused_scores.get(doc_id, 0.0)}
                for (doc_id, doc), score in zip(rerank_pool, rerank_scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        top_docs = [item["doc"] for item in reranked[:top_k]]

        debug = {
            "vector_hits": [
                {
                    "doc_id": doc.metadata["doc_id"],
                    "score": float(score),
                    "source_id": doc.metadata.get("source_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
                for doc, score in vector_hits
            ],
            "bm25_hits": [
                {
                    "doc_id": doc.metadata["doc_id"],
                    "score": bm25_scores.get(doc.metadata["doc_id"], 0.0),
                    "source_id": doc.metadata.get("source_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
                for doc in bm25_docs.values()
            ],
            "fused": [
                {"doc_id": doc_id, "score": float(score)}
                for doc_id, score in sorted_candidates[: max(1, top_k * 3)]
            ],
            "rerank": [
                {
                    "doc_id": item["doc_id"],
                    "rerank_score": item["score"],
                    "fused_score": item["fused"],
                }
                for item in reranked
            ],
        }

        return top_docs, debug

    def generate_answer(
        self, query: str, top_k: int = 4, alpha: float = 0.5
    ) -> Tuple[str, List[Document], Dict[str, List[Dict[str, float]]]]:
        if not self.is_ready():
            raise ValueError("Indexes not built. Ingest text and call build_indexes().")

        top_docs, debug = self.retrieve(query=query, top_k=top_k, alpha=alpha)
        context = "\n\n".join(
            [f"[Doc {i}] {doc.page_content}" for i, doc in enumerate(top_docs, start=1)]
        )

        system_prompt = (
            "You are a precise assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not present, say you do not know."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        if self.llm is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided.")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.0,
                openai_api_key=self.openai_api_key,
            )

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        )

        return response.content, top_docs, debug

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _normalize_scores(scores: Dict[int, float], method: str = "minmax") -> Dict[int, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=np.float32)
        if method == "zscore":
            mean, std = float(values.mean()), float(values.std())
            if std == 0:
                return {k: 0.0 for k in scores}
            return {k: float((v - mean) / std) for k, v in scores.items()}
        # default min-max
        min_s, max_s = float(values.min()), float(values.max())
        if max_s == min_s:
            return {k: 1.0 for k in scores}
        return {k: float((v - min_s) / (max_s - min_s)) for k, v in scores.items()}

    @staticmethod
    def _rrf_fuse(vec_scores: Dict[int, float], bm25_scores: Dict[int, float], k: int = 60) -> Dict[int, float]:
        """Reciprocal Rank Fusion: rank-based and less sensitive to score scale."""
        fused: Dict[int, float] = {}
        # ranks are zero-based; add 1 to avoid div by zero.
        vec_ranked = [doc_id for doc_id, _ in sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)]
        bm25_ranked = [doc_id for doc_id, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)]
        for rank, doc_id in enumerate(vec_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(bm25_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused
