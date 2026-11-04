import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from supabase import create_client, Client

load_dotenv()

DEFAULT_BUCKET_NAME = "research-papers"


def get_supabase_client() -> Optional[Client]:
    """
    إنشاء وإرجاع كائن Supabase Client باستخدام متغيرات البيئة.
    يدعم SUPABASE_SERVICE_ROLE_KEY أو SUPABASE_KEY مع تنظيف الرابط تلقائياً.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        return None

    # تنظيف الرابط في حال تم لصق /rest/v1/ في نهايته
    clean_url = supabase_url.split("/rest/v1")[0].rstrip("/")

    return create_client(clean_url, supabase_key)


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    حساب الـ SHA-256 Hash لمحتوى الملف لضمان عدم التكرار (Deduplication).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_hash(file_bytes: bytes) -> str:
    """
    حساب الـ SHA-256 Hash مباشرة من مصفوفة الـ Bytes.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def ensure_storage_bucket(client: Client, bucket_name: str = DEFAULT_BUCKET_NAME) -> None:
    """
    التأكد من وجود الـ Storage Bucket في Supabase وإنشائه إن لم يكن موجوداً.
    """
    try:
        buckets = client.storage.list_buckets()
        existing_names = [b.name for b in buckets]
        if bucket_name not in existing_names:
            client.storage.create_bucket(bucket_name, options={"public": False})
            print(f"[Supabase Storage] Created bucket '{bucket_name}'.")
    except Exception:
        pass


def _normalize_doc_record(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    توحيد تسميات الحقول بين schemas المختلفة (file_id/id و file_name/filename).
    """
    if not doc:
        return doc
    
    # ضمان توفر كلا المفتاحين id و file_id
    if "file_id" in doc and "id" not in doc:
        doc["id"] = doc["file_id"]
    elif "id" in doc and "file_id" not in doc:
        doc["file_id"] = doc["id"]

    # ضمان توفر كلا المفتاحين filename و file_name
    if "file_name" in doc and "filename" not in doc:
        doc["filename"] = doc["file_name"]
    elif "filename" in doc and "file_name" not in doc:
        doc["file_name"] = doc["filename"]

    return doc


def get_document_by_hash(client: Client, file_hash: str) -> Optional[Dict[str, Any]]:
    """
    البحث عن مستند في جدول documents باستخدام الـ file_hash.
    """
    response = client.table("documents").select("*").eq("file_hash", file_hash).execute()
    data = response.data
    return _normalize_doc_record(data[0]) if data and len(data) > 0 else None


def get_document_by_id(client: Client, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    البحث عن مستند في جدول documents باستخدام الـ UUID.
    يدعم البحث عبر عمود file_id أو id.
    """
    try:
        response = client.table("documents").select("*").eq("file_id", doc_id).execute()
        if response.data and len(response.data) > 0:
            return _normalize_doc_record(response.data[0])
    except Exception:
        pass

    try:
        response = client.table("documents").select("*").eq("id", doc_id).execute()
        if response.data and len(response.data) > 0:
            return _normalize_doc_record(response.data[0])
    except Exception:
        pass

    return None


def list_documents(client: Client, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    جلب قائمة بجميع المستندات المسجلة مرتبة بالأحدث.
    """
    response = (
        client.table("documents")
        .select("*")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    docs = response.data or []
    return [_normalize_doc_record(d) for d in docs]


def upload_pdf_to_storage(
    client: Client,
    file_path: Union[str, Path],
    file_hash: str,
    bucket_name: str = DEFAULT_BUCKET_NAME
) -> str:
    """
    رفع ملف الـ PDF من مسار محلي إلى Supabase Storage Bucket باسم <file_hash>.pdf
    """
    path = Path(file_path)
    storage_path = f"{file_hash}.pdf"

    ensure_storage_bucket(client, bucket_name=bucket_name)

    with open(path, "rb") as f:
        file_bytes = f.read()

    client.storage.from_(bucket_name).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"}
    )
    return storage_path


def upload_bytes_to_storage(
    client: Client,
    file_bytes: bytes,
    file_hash: str,
    bucket_name: str = DEFAULT_BUCKET_NAME
) -> str:
    """
    رفع محتوى الـ PDF مباشرة من الـ Bytes إلى Supabase Storage Bucket.
    """
    storage_path = f"{file_hash}.pdf"
    ensure_storage_bucket(client, bucket_name=bucket_name)

    client.storage.from_(bucket_name).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"}
    )
    return storage_path


def delete_file_from_storage(
    client: Client,
    storage_path: str,
    bucket_name: str = DEFAULT_BUCKET_NAME
) -> None:
    """
    حذف ملف من الـ Supabase Storage Bucket.
    """
    try:
        client.storage.from_(bucket_name).remove([storage_path])
    except Exception:
        pass


def delete_document_record(client: Client, doc_id: str) -> None:
    """
    حذف سجل المستند من جدول documents في قاعدة البيانات.
    """
    try:
        client.table("documents").delete().eq("file_id", doc_id).execute()
    except Exception:
        pass
    try:
        client.table("documents").delete().eq("id", doc_id).execute()
    except Exception:
        pass


def create_or_update_document_record(
    client: Client,
    filename: str,
    file_hash: str,
    storage_path: str,
    file_size: int = 0
) -> Dict[str, Any]:
    """
    إنشاء سجل جديد أو تحديث السجل بحالة 'processing'.
    يدعم الـ schema الفعلي (file_name, file_size, file_id) مع التوافق التلقائي.
    """
    existing = get_document_by_hash(client, file_hash)

    payload = {
        "file_hash": file_hash,
        "file_name": filename,
        "file_size": file_size,
        "storage_path": storage_path,
        "status": "processing",
        "error_message": None,
    }

    try:
        if existing:
            doc_id = existing.get("file_id") or existing.get("id")
            try:
                res = client.table("documents").update(payload).eq("file_id", doc_id).execute()
            except Exception:
                res = client.table("documents").update(payload).eq("id", doc_id).execute()
            return _normalize_doc_record(res.data[0]) if res.data else existing
        else:
            res = client.table("documents").insert(payload).execute()
            return _normalize_doc_record(res.data[0])
    except Exception:
        # Fallback في حال كان الجدول يستخدم أعمدة قديمة
        legacy_payload = {
            "file_hash": file_hash,
            "filename": filename,
            "storage_path": storage_path,
            "status": "processing",
            "error_message": None,
        }
        if existing:
            doc_id = existing.get("id") or existing.get("file_id")
            res = client.table("documents").update(legacy_payload).eq("id", doc_id).execute()
            return _normalize_doc_record(res.data[0]) if res.data else existing
        else:
            res = client.table("documents").insert(legacy_payload).execute()
            return _normalize_doc_record(res.data[0])


def mark_document_indexed(client: Client, doc_id: str, total_chunks: int) -> Dict[str, Any]:
    """
    تحديث حالة المستند إلى 'indexed' مع تسجيل عدد الـ chunks.
    """
    payload = {
        "status": "indexed",
        "total_chunks": total_chunks,
        "error_message": None,
    }
    try:
        res = client.table("documents").update(payload).eq("file_id", doc_id).execute()
        if res.data:
            return _normalize_doc_record(res.data[0])
    except Exception:
        pass

    try:
        res = client.table("documents").update(payload).eq("id", doc_id).execute()
        if res.data:
            return _normalize_doc_record(res.data[0])
    except Exception:
        pass

    return payload


def mark_document_failed(client: Client, doc_id: str, error_message: str) -> Dict[str, Any]:
    """
    تحديث حالة المستند إلى 'failed' مع تسجيل سبب الخطأ.
    """
    payload = {
        "status": "failed",
        "error_message": error_message[:1000] if error_message else "Unknown error",
    }
    try:
        res = client.table("documents").update(payload).eq("file_id", doc_id).execute()
        if res.data:
            return _normalize_doc_record(res.data[0])
    except Exception:
        pass

    try:
        res = client.table("documents").update(payload).eq("id", doc_id).execute()
        if res.data:
            return _normalize_doc_record(res.data[0])
    except Exception:
        pass

    return payload
