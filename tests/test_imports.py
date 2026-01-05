import importlib


def test_core_imports() -> None:
    # Verify key libraries resolve (catches version pin issues like cached_download removal).
    importlib.import_module("sentence_transformers")
    importlib.import_module("huggingface_hub")
    importlib.import_module("faiss")
    importlib.import_module("rank_bm25")
    importlib.import_module("langchain")


def test_engine_instantiation() -> None:
    # Ensure the engine module loads without ImportError.
    from rag_engine import AdvancedRAGEngine

    eng = AdvancedRAGEngine(openai_api_key="dummy")
    assert eng is not None
