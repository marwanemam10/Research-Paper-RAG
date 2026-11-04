import os
import sys
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# ضبط الـ encoding لـ UTF-8 على Windows
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingestion.ingestion_service import IngestionService
from src.retrieval.retriever import get_retriever, retrieve_documents
from src.generation.generation import generate_answer
from src.storage.supabase_client import (
    get_supabase_client,
    list_documents,
    get_document_by_id,
    DEFAULT_BUCKET_NAME,
)
from src.storage.vector_store import get_qdrant_client, DEFAULT_COLLECTION_NAME

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Academic Research Paper RAG API",
    description="Decoupled Production RAG API using FastAPI, Supabase (Postgres & Storage), Qdrant, and Google Gemini.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# السماح بالـ CORS للواجهات الأمامية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="السؤال البحثي المراد الاستعلام عنه")
    k: int = Field(default=4, ge=1, le=20, description="عدد الـ Chunks المسترجعة للسياق")
    session_id: Optional[str] = Field(default="api_session", description="معرف الجلسة لتتبعها في Langfuse")
    user_id: Optional[str] = Field(default="api_user", description="معرف المستخدم")


class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    file_id: Optional[str] = None
    content_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    session_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="نص البحث الدلالي")
    k: int = Field(default=4, ge=1, le=20, description="عدد النتائج")


class SyncLocalRequest(BaseModel):
    directory: str = Field(default="src/data", description="مسار المجلد المحلي الذي يحتوي على ملفات الـ PDF")
    force: bool = Field(default=False, description="إعادة الفهرسة حتى لو كانت الملفات مسجلة مسبقاً")


# ==============================================================================
# 1. Health & Status Endpoints
# ==============================================================================

@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Academic Research Paper RAG API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/api/v1/health"
    }


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """
    فحص حالة اتصال الخدمات: Supabase, Qdrant, و Gemini API.
    """
    health = {
        "status": "healthy",
        "supabase": {"connected": False},
        "qdrant": {"connected": False},
        "gemini_api": {"configured": bool(os.getenv("GEMINI_API_KEY"))}
    }

    # فحص Supabase
    supabase = get_supabase_client()
    if supabase:
        try:
            buckets = supabase.storage.list_buckets()
            doc_count = len(list_documents(supabase, limit=1))
            health["supabase"] = {
                "connected": True,
                "bucket": DEFAULT_BUCKET_NAME,
                "bucket_exists": any(b.name == DEFAULT_BUCKET_NAME for b in buckets),
                "has_documents": doc_count >= 0
            }
        except Exception as e:
            health["supabase"] = {"connected": False, "error": str(e)}
            health["status"] = "degraded"
    else:
        health["supabase"] = {"connected": False, "error": "Credentials missing in .env"}

    # فحص Qdrant
    qdrant = get_qdrant_client()
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        has_collection = DEFAULT_COLLECTION_NAME in collections
        points = qdrant.count(DEFAULT_COLLECTION_NAME).count if has_collection else 0
        health["qdrant"] = {
            "connected": True,
            "collection": DEFAULT_COLLECTION_NAME,
            "collection_exists": has_collection,
            "total_points": points
        }
    except Exception as e:
        health["qdrant"] = {"connected": False, "error": str(e)}
        health["status"] = "degraded"

    return health


# ==============================================================================
# 2. Ingestion Endpoints (Supabase & Qdrant)
# ==============================================================================

@app.post("/api/v1/documents/upload", tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(..., description="ملف البحث بصيغة PDF"),
    force: bool = Form(default=False, description="إعادة المعالجة والفهرسة حتى لو كان الملف مسجلاً")
):
    """
    رفع ملف PDF مباشرة إلى مسار الـ Ingestion:
    1. حساب الـ SHA-256 Hash.
    2. التحقق من التكرار عبر Supabase.
    3. رفع الملف إلى Supabase Storage.
    4. التقطيع وحقن الـ file_id في الـ metadata.
    5. التضمين والفهرسة في Qdrant.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="فقط ملفات الـ PDF مدعومة في هذه الخدمة."
        )

    try:
        content = await file.read()
        service = IngestionService(collection_name=DEFAULT_COLLECTION_NAME)
        result = service.ingest_bytes(
            filename=file.filename,
            file_bytes=content,
            force=force
        )
        return {
            "message": "تمت معالجة الملف بنجاح",
            "data": result
        }
    except Exception as e:
        logger.error(f"Error ingesting uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشلت عملية معالجة الملف: {str(e)}"
        )


@app.post("/api/v1/documents/sync-local", tags=["Ingestion"])
async def sync_local_directory(req: SyncLocalRequest):
    """
    مزامنة وفهرسة ملفات الـ PDF الموجودة في مجلد محلي إلى Supabase و Qdrant.
    """
    target_path = Path(req.directory)
    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المجلد غير موجود: {target_path}"
        )

    service = IngestionService(collection_name=DEFAULT_COLLECTION_NAME)
    results = service.ingest_directory(target_path, force=req.force)
    return {
        "message": f"اكتملت مزامنة {len(results)} ملف(ات)",
        "results": results
    }


@app.get("/api/v1/documents", tags=["Ingestion"])
async def get_all_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    جلب قائمة بجميع المستندات المسجلة في جدول Supabase documents.
    """
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase credentials not configured in .env"
        )

    docs = list_documents(supabase, limit=limit, offset=offset)
    return {
        "count": len(docs),
        "limit": limit,
        "offset": offset,
        "documents": docs
    }


@app.get("/api/v1/documents/{doc_id}", tags=["Ingestion"])
async def get_document_details(doc_id: str):
    """
    جلب بيانات مستند معين بواسطة الـ UUID.
    """
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase credentials not configured"
        )

    doc = get_document_by_id(supabase, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستند غير موجود: {doc_id}"
        )
    return doc


@app.delete("/api/v1/documents/{doc_id}", tags=["Ingestion"])
async def delete_document(doc_id: str):
    """
    حذف المستند نهائياً من Qdrant، و Supabase Storage، وسجل جدول documents.
    """
    try:
        service = IngestionService(collection_name=DEFAULT_COLLECTION_NAME)
        res = service.delete_document(doc_id)
        if res.get("status") == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"المستند غير موجود: {doc_id}"
            )
        return {
            "message": "تم حذف المستند ومتجهاته وملفه بنجاح",
            "data": res
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء الحذف: {str(e)}"
        )


# ==============================================================================
# 3. Retrieval & Query Endpoints
# ==============================================================================

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_rag(req: QueryRequest):
    """
    تنفيذ استعلام كامل: استرجاع السياق الأكثر صلة من Qdrant وتوليد إجابة أكاديمية موثقة عبر Gemini.
    """
    qdrant = get_qdrant_client()
    try:
        if not qdrant.collection_exists(DEFAULT_COLLECTION_NAME):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"المجموعة '{DEFAULT_COLLECTION_NAME}' غير موجودة. يرجى فهرسة بعض المستندات أولاً."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"تعذر الاتصال بقاعدة بيانات Qdrant: {str(e)}"
        )

    try:
        retriever = get_retriever(collection_name=DEFAULT_COLLECTION_NAME, k=req.k)

        # 1. استرجاع المستندات ذات الصلة
        retrieved_docs = retrieve_documents(
            query=req.question,
            retriever=retriever,
            session_id=req.session_id,
            user_id=req.user_id
        )

        # 2. توليد الإجابة
        answer = generate_answer(
            question=req.question,
            retriever=retriever,
            session_id=req.session_id,
            user_id=req.user_id
        )

        # 3. إعداد المصادر
        sources: List[SourceItem] = []
        for doc in retrieved_docs:
            src_path = doc.metadata.get("source", "Unknown")
            src_name = Path(src_path).name if src_path else "Unknown"
            page = doc.metadata.get("page", None)
            file_id = doc.metadata.get("file_id", None)
            preview = doc.page_content.strip().replace("\n", " ")[:200] + "..."

            sources.append(
                SourceItem(
                    source=src_name,
                    page=page + 1 if page is not None else None,
                    file_id=file_id,
                    content_preview=preview
                )
            )

        return QueryResponse(
            question=req.question,
            answer=answer,
            sources=sources,
            session_id=req.session_id
        )

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء المعالجة وتوليد الإجابة: {str(e)}"
        )


@app.post("/api/v1/retrieve", tags=["Retrieval"])
async def retrieve_only(req: RetrieveRequest):
    """
    استرجاع المقاطع (Chunks) ذات الصلة بالسؤال فقط دون استدعاء نموذج التوليد LLM.
    """
    try:
        retriever = get_retriever(collection_name=DEFAULT_COLLECTION_NAME, k=req.k)
        docs = retriever.invoke(req.query)

        results = []
        for d in docs:
            results.append({
                "content": d.page_content,
                "metadata": d.metadata
            })
        return {
            "query": req.query,
            "count": len(results),
            "chunks": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل الاسترجاع: {str(e)}"
        )


# ==============================================================================
# Main Runner
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Academic Research RAG FastAPI Server...")
    print("📖 Swagger Documentation: http://localhost:8000/docs")
    print("📘 ReDoc Documentation:   http://localhost:8000/redoc\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

