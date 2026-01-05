import textwrap
from typing import Dict, List

import plotly.graph_objects as go

import streamlit as st

from rag_engine import AdvancedRAGEngine


st.set_page_config(
    page_title="Advanced Hybrid RAG",
    page_icon="🧭",
    layout="wide",
)

DEFAULT_KB = textwrap.dedent(
    """
    Retrieval-Augmented Generation (RAG) pairs document retrieval with a language model:
    - Dense retrieval relies on vector similarity for semantic matches.
    - Sparse retrieval (BM25) keeps lexical precision and exact term recall.
    - Hybrid search fuses both to mitigate weaknesses of each method.
    - Cross-encoder re-ranking improves ordering by scoring query-document pairs jointly.
    - FAISS HNSW indexes use graph-based search to deliver fast approximate nearest neighbors with strong recall.
    """
).strip()

if "engine" not in st.session_state:
    st.session_state.engine = None
if "history" not in st.session_state:
    st.session_state.history = []


def build_sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.title("Controls")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Key is used only in-session for generation calls.",
        )
        alpha = st.slider(
            "Hybrid weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0 = keyword heavy, 1 = vector heavy.",
        )
        kb_text = st.text_area(
            "Upload or paste knowledge base text",
            value=DEFAULT_KB,
            height=260,
        )

        if st.button("Build Index", type="primary"):
            if not kb_text.strip():
                st.warning("Please provide some content before building the index.")
            else:
                engine = AdvancedRAGEngine(openai_api_key=api_key)
                engine.ingest_texts([kb_text])
                engine.build_indexes()
                st.session_state.engine = engine
                st.session_state.history = []
                st.success("Index built. Start asking questions in the chat.")

        return {"api_key": api_key, "alpha": alpha}


def render_sources(sources: List[Dict[str, str]]) -> None:
    for idx, src in enumerate(sources, start=1):
        st.markdown(f"**Source {idx}** (doc {src['doc_id']}): {src['content']}")


def _plot_stage_scores(title: str, records: List[Dict], score_key: str) -> None:
    if not records:
        return
    sorted_records = sorted(records, key=lambda r: r.get(score_key, 0.0), reverse=True)
    labels = [
        f"doc {r.get('doc_id', '-')}/c{r.get('chunk_id', '-')}"
        if r.get("chunk_id") is not None
        else f"doc {r.get('doc_id', '-')}"
        for r in sorted_records
    ]
    scores = [r.get(score_key, 0.0) for r in sorted_records]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
        )
    )
    fig.update_layout(title=title, yaxis_title=score_key, xaxis_title="doc/chunk")
    st.plotly_chart(fig, use_container_width=True)


def _plot_rerank(records: List[Dict]) -> None:
    if not records:
        return
    sorted_records = sorted(records, key=lambda r: r.get("rerank_score", 0.0), reverse=True)
    labels = [f"doc {r.get('doc_id', '-')}" for r in sorted_records]
    rerank_scores = [r.get("rerank_score", 0.0) for r in sorted_records]
    fused_scores = [r.get("fused_score", 0.0) for r in sorted_records]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=rerank_scores,
            name="rerank",
            text=[f"{s:.3f}" for s in rerank_scores],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fused_scores,
            name="fused (pre-rerank)",
            text=[f"{s:.3f}" for s in fused_scores],
            textposition="auto",
        )
    )
    fig.update_layout(barmode="group", title="Rerank vs fused scores", yaxis_title="score")
    st.plotly_chart(fig, use_container_width=True)


def render_debug_charts(debug: Dict) -> None:
    st.markdown("**Score charts**")
    _plot_stage_scores("Vector scores", debug.get("vector_hits", []), "score")
    _plot_stage_scores("BM25 scores", debug.get("bm25_hits", []), "score")
    _plot_stage_scores("Fused scores (pre rerank)", debug.get("fused", []), "score")
    _plot_rerank(debug.get("rerank", []))


def main() -> None:
    controls = build_sidebar()
    st.title("Advanced Hybrid RAG Demo")
    st.caption(
        "Production-grade hybrid retrieval with FAISS HNSW + BM25, cross-encoder re-ranking, and transparent debugging."
    )

    engine: AdvancedRAGEngine = st.session_state.engine

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = message.get("sources", [])
                if sources:
                    st.markdown("**Sources**")
                    render_sources(sources)
                debug = message.get("debug")
                if debug:
                    with st.expander("Debug view"):
                        st.markdown("**Vector hits**")
                        st.json(debug.get("vector_hits", []))
                        st.markdown("**BM25 hits**")
                        st.json(debug.get("bm25_hits", []))
                        st.markdown("**Fused scores (pre re-rank)**")
                        st.json(debug.get("fused", []))
                        st.markdown("**Cross-encoder re-rank scores**")
                        st.json(debug.get("rerank", []))
                        render_debug_charts(debug)

    prompt = st.chat_input("Ask about your knowledge base")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        if engine is None:
            st.warning("Please build the index first from the sidebar.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    try:
                        answer, docs, debug = engine.generate_answer(
                            query=prompt,
                            top_k=4,
                            alpha=controls["alpha"],
                        )
                    except Exception as exc:  # pragma: no cover - surface to user
                        st.error(f"Error: {exc}")
                        return

                st.markdown(answer)
                source_payload = [
                    {
                        "doc_id": doc.metadata.get("doc_id"),
                        "source_id": doc.metadata.get("source_id"),
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "content": doc.page_content,
                    }
                    for doc in docs
                ]
                if source_payload:
                    st.markdown("**Sources**")
                    render_sources(source_payload)
                with st.expander("Debug view"):
                    st.markdown("**Vector hits**")
                    st.json(debug.get("vector_hits", []))
                    st.markdown("**BM25 hits**")
                    st.json(debug.get("bm25_hits", []))
                    st.markdown("**Fused scores (pre re-rank)**")
                    st.json(debug.get("fused", []))
                    st.markdown("**Cross-encoder re-rank scores**")
                    st.json(debug.get("rerank", []))
                    render_debug_charts(debug)

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_payload,
                        "debug": debug,
                    }
                )


if __name__ == "__main__":
    main()
