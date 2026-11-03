import os
from typing import List, Optional
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from langchain_core.documents import Document
# pyrefly: ignore[missing-import]
from langchain_qdrant import QdrantVectorStore
# pyrefly: ignore[missing-import]
from langfuse.langchain import CallbackHandler

from src.storage.vector_store import get_vector_store, DEFAULT_COLLECTION_NAME

load_dotenv()


def get_retriever(
    vector_store: Optional[QdrantVectorStore] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 4,
    search_type: str = "similarity"
):
    """
    إنشاء كائن الـ Retriever مباشرة من Qdrant VectorStore دون أي عمليات فحص للملفات.
    """
    if vector_store is None:
        vector_store = get_vector_store(collection_name=collection_name)

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )


def retrieve_documents(
    query: str,
    retriever=None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> List[Document]:
    """
    تنفيذ الاسترجاع مع تسجيل الـ Traces تلقائياً في Langfuse.
    """
    if retriever is None:
        retriever = get_retriever()

    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "query": query,
            "session_id": session_id or "default_session",
            "user_id": user_id or "guest_user"
        }
    }

    docs = retriever.invoke(query, config=config)
    return docs
