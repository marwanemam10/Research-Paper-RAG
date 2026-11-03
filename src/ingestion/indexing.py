import os
import time
from typing import List, Optional
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from langchain_core.documents import Document
# pyrefly: ignore[missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore[missing-import]
from qdrant_client.http.models import Distance, VectorParams
# pyrefly: ignore[missing-import]
from langchain_qdrant import QdrantVectorStore

from src.models.models import embd
from src.storage.vector_store import get_qdrant_client, get_vector_store

load_dotenv()

DEFAULT_COLLECTION_NAME = "research_papers"


def index_documents(
    documents: List[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: Optional[QdrantClient] = None,
    embedding=None,
    recreate_collection: bool = False,
    batch_size: int = 50,
) -> Optional[QdrantVectorStore]:
    """
    تخزين المستندات (Chunks) داخل Qdrant وإنشاء الـ Index مع إدارة الـ Rate Limits.
    
    Args:
        documents: قائمة بالـ Chunks (Document objects) الناتجة عن عملية الـ splitting.
        collection_name: اسم الـ collection في Qdrant.
        client: كائن QdrantClient (اختياري).
        embedding: موديل الـ Embedding المستخدم لتوليد المتجهات (افتراضياً embd).
        recreate_collection: إذا كان True، سيتم مسح الـ collection السابقة والبدء من جديد.
        batch_size: حجم الدفعة الواحدة لتفادي تجاوز حد الـ Rate Limit لـ API التضمين المجاني.
        
    Returns:
        QdrantVectorStore: كائن الـ vectorstore بعد الفهرسة.
    """
    if not documents:
        print("[Warning] Document list is empty. No indexing performed.")
        return None

    if client is None:
        client = get_qdrant_client()

    if embedding is None:
        embedding = embd

    # إذا طُلب مسح الـ collection السابقة وإعادة إنشائها
    if recreate_collection and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[Info] Deleted existing collection: '{collection_name}'")

    # إنشاء الـ collection إذا لم تكن موجودة بالفعل
    if not client.collection_exists(collection_name):
        sample_vec = embedding.embed_query("test")
        vector_size = len(sample_vec)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"[Info] Created new collection: '{collection_name}' with vector size: {vector_size}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    total_docs = len(documents)
    print(f"[Info] Indexing {total_docs} chunks into Qdrant collection '{collection_name}' (batches of {batch_size})...")

    # رفع المستندات على دفعات مع معالجة حد الـ Rate Limit وإعادة المحاولة التلقائية
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        print(f"  -> Indexing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch, batch_size=batch_size)
                time.sleep(1)
                break
            except Exception as e:
                err_msg = str(e)
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    wait_seconds = 20 * (attempt + 1)
                    print(f"     [Rate Limit] Free tier limit reached. Waiting {wait_seconds}s before retrying...")
                    time.sleep(wait_seconds)
                else:
                    raise e

    print(f"[Success] All {total_docs} chunks indexed successfully in '{collection_name}'.")
    return vector_store
