import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from langchain_core.documents import Document

from src.ingestion.splitting import split_pdf
from src.ingestion.indexing import index_documents
from src.storage.supabase_client import (
    get_supabase_client,
    compute_file_hash,
    compute_bytes_hash,
    get_document_by_hash,
    get_document_by_id,
    upload_pdf_to_storage,
    upload_bytes_to_storage,
    delete_file_from_storage,
    delete_document_record,
    create_or_update_document_record,
    mark_document_indexed,
    mark_document_failed,
    DEFAULT_BUCKET_NAME
)
from src.storage.vector_store import get_qdrant_client, delete_documents_by_file_id

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class IngestionService:
    """
    خدمة معالجة وفهرسة المستندات المستقلة:
    1. حساب الـ Hash ومنع تكرار الملفات (Deduplication).
    2. رفع الملفات إلى Supabase Storage وتسجيل حالتها في جدول documents.
    3. تقطيع المستند وحقن الـ file_id و source في الـ metadata.
    4. التضمين والفهرسة في Qdrant.
    5. تحديث حالة المستند إلى 'indexed' أو 'failed'.
    """

    def __init__(self, collection_name: str = "research_papers"):
        self.collection_name = collection_name
        self.supabase = get_supabase_client()
        self.qdrant = get_qdrant_client()

    def ingest_bytes(
        self,
        filename: str,
        file_bytes: bytes,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        معالجة محتوى PDF تم رفعه مباشرة عبر الـ API كـ Bytes.
        """
        file_hash = compute_bytes_hash(file_bytes)
        print(f"\n>> Ingesting Bytes for '{filename}' (SHA-256: {file_hash[:12]}...)")

        # التحقق من وجود المستند مسبقاً في Supabase
        if self.supabase:
            existing_doc = get_document_by_hash(self.supabase, file_hash)
            if existing_doc and existing_doc.get("status") == "indexed" and not force:
                print(f"[Deduplication] '{filename}' is already indexed. Skipping.")
                doc_id_val = existing_doc.get("file_id") or existing_doc.get("id")
                return {
                    "status": "already_indexed",
                    "doc_id": doc_id_val,
                    "filename": filename,
                    "file_hash": file_hash,
                    "total_chunks": existing_doc.get("total_chunks", 0),
                    "storage_path": existing_doc.get("storage_path")
                }

            if existing_doc and force:
                old_id = existing_doc.get("file_id") or existing_doc.get("id")
                delete_documents_by_file_id(file_id=old_id, collection_name=self.collection_name, client=self.qdrant)

        # حفظ الملف مؤقتاً لتقطيعه بواسطة PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        doc_id = None
        try:
            storage_path = f"{file_hash}.pdf"
            if self.supabase:
                print(f"[Storage] Uploading to Supabase Storage '{DEFAULT_BUCKET_NAME}/{storage_path}'...")
                storage_path = upload_bytes_to_storage(
                    client=self.supabase,
                    file_bytes=file_bytes,
                    file_hash=file_hash,
                    bucket_name=DEFAULT_BUCKET_NAME
                )

                doc_record = create_or_update_document_record(
                    client=self.supabase,
                    filename=filename,
                    file_hash=file_hash,
                    storage_path=storage_path,
                    file_size=len(file_bytes)
                )
                doc_id = doc_record.get("file_id") or doc_record.get("id")

            # التقطيع
            chunks = split_pdf(tmp_path)
            total_chunks = len(chunks)

            # حقن الـ Metadata
            for chunk in chunks:
                if doc_id:
                    chunk.metadata["file_id"] = str(doc_id)
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["source"] = filename

            # الفهرسة في Qdrant
            index_documents(
                documents=chunks,
                collection_name=self.collection_name,
                client=self.qdrant,
                recreate_collection=False
            )

            if doc_id and self.supabase:
                mark_document_indexed(client=self.supabase, doc_id=doc_id, total_chunks=total_chunks)

            return {
                "status": "indexed",
                "doc_id": doc_id,
                "filename": filename,
                "file_hash": file_hash,
                "total_chunks": total_chunks,
                "storage_path": storage_path
            }

        except Exception as e:
            if doc_id and self.supabase:
                mark_document_failed(client=self.supabase, doc_id=doc_id, error_message=str(e))
            raise e
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def ingest_file(
        self,
        file_path: Union[str, Path],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        معالجة ملف PDF واحد من مسار محلي على القرص.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            content = f.read()

        return self.ingest_bytes(filename=path.name, file_bytes=content, force=force)

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        معالجة جميع ملفات الـ PDF الموجودة في مجلد معين.
        """
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        pdf_files = sorted(list(path.glob("*.pdf")))
        if not pdf_files:
            print(f"[Info] No PDF files found in '{path}'.")
            return []

        print(f"\n==================================================")
        print(f"🚀 Starting Ingestion Pipeline for {len(pdf_files)} PDF(s) in '{path}'")
        print(f"==================================================")

        results = []
        for pdf_file in pdf_files:
            try:
                res = self.ingest_file(pdf_file, force=force)
                results.append(res)
            except Exception as e:
                print(f"[Failure] Error processing {pdf_file.name}: {e}")
                results.append({"status": "failed", "filename": pdf_file.name, "error": str(e)})

        print(f"\n==================================================")
        print(f"✅ Ingestion Batch Complete ({len(results)} processed)")
        print(f"==================================================")
        return results

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        حذف المستند بالكامل من Qdrant، و Supabase Storage، وسجلات قاعدة البيانات.
        """
        if not self.supabase:
            raise RuntimeError("Supabase client is not configured.")

        doc = get_document_by_id(self.supabase, doc_id)
        if not doc:
            return {"status": "not_found", "doc_id": doc_id}

        # 1. حذف المتجهات من Qdrant
        delete_documents_by_file_id(file_id=doc_id, collection_name=self.collection_name, client=self.qdrant)

        # 2. حذف الملف من Supabase Storage
        storage_path = doc.get("storage_path")
        if storage_path:
            delete_file_from_storage(self.supabase, storage_path=storage_path)

        # 3. حذف السجل من جدول documents
        delete_document_record(self.supabase, doc_id=doc_id)

        return {"status": "deleted", "doc_id": doc_id, "filename": doc.get("filename")}
