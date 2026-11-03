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


def get_document_by_hash(client: Client, file_hash: str) -> Optional[Dict[str, Any]]:
    """
    البحث عن مستند في جدول documents باستخدام الـ file_hash.
    """
    response = client.table("documents").select("*").eq("file_hash", file_hash).execute()
    data = response.data
    return data[0] if data and len(data) > 0 else None


def get_document_by_id(client: Client, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    البحث عن مستند في جدول documents باستخدام الـ UUID.
    """
    response = client.table("documents").select("*").eq("id", doc_id).execute()
    data = response.data
    return data[0] if data and len(data) > 0 else None


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
    return response.data or []


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
    client.table("documents").delete().eq("id", doc_id).execute()


def create_or_update_document_record(
    client: Client,
    filename: str,
    file_hash: str,
    storage_path: str
) -> Dict[str, Any]:
    """
    إنشاء سجل جديد أو تحديث السجل بحالة 'processing'.
    """
    existing = get_document_by_hash(client, file_hash)

    payload = {
        "file_hash": file_hash,
        "filename": filename,
        "storage_path": storage_path,
        "status": "processing",
        "error_message": None,
    }

    if existing:
        doc_id = existing["id"]
        response = client.table("documents").update(payload).eq("id", doc_id).execute()
        return response.data[0]
    else:
        response = client.table("documents").insert(payload).execute()
        return response.data[0]


def mark_document_indexed(client: Client, doc_id: str, total_chunks: int) -> Dict[str, Any]:
    """
    تحديث حالة المستند إلى 'indexed' مع تسجيل عدد الـ chunks.
    """
    payload = {
        "status": "indexed",
        "total_chunks": total_chunks,
        "error_message": None,
    }
    response = client.table("documents").update(payload).eq("id", doc_id).execute()
    return response.data[0] if response.data else payload


def mark_document_failed(client: Client, doc_id: str, error_message: str) -> Dict[str, Any]:
    """
    تحديث حالة المستند إلى 'failed' مع تسجيل سبب الخطأ.
    """
    payload = {
        "status": "failed",
        "error_message": error_message[:1000] if error_message else "Unknown error",
    }
    response = client.table("documents").update(payload).eq("id", doc_id).execute()
    return response.data[0] if response.data else payload
