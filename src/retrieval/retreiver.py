# Re-exporting from retriever.py for backward compatibility
from src.retrieval.retriever import get_retriever, retrieve_documents

__all__ = ["get_retriever", "retrieve_documents"]
