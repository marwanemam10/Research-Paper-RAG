import os
from typing import Optional
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore[missing-import]
from qdrant_client.http import models
# pyrefly: ignore[missing-import]
from langchain_qdrant import QdrantVectorStore

from src.models.models import embd

load_dotenv()

DEFAULT_COLLECTION_NAME = "research_papers"


def get_qdrant_client(
    url: Optional[str] = None, 
    api_key: Optional[str] = None
) -> QdrantClient:
    """
    إنشاء وإرجاع كائن الاتصال بقاعدة بيانات Qdrant
    """
    target_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    target_api_key = api_key or os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=target_url, api_key=target_api_key)


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: Optional[QdrantClient] = None,
    embedding=None,
) -> QdrantVectorStore:
    """
    إرجاع كائن QdrantVectorStore متصل بالمجموعة للبحث والاسترجاع.
    """
    if client is None:
        client = get_qdrant_client()
    if embedding is None:
        embedding = embd

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )


def delete_documents_by_file_id(
    file_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: Optional[QdrantClient] = None
) -> None:
    """
    حذف جميع الـ Chunks المرتبطة بـ file_id محدد من مجموعة Qdrant.
    """
    if client is None:
        client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        return

    # حذف النقاط التي تطابق metadata.file_id
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=file_id),
                    ),
                    models.FieldCondition(
                        key="file_id",
                        match=models.MatchValue(value=file_id),
                    ),
                ]
            )
        ),
    )
