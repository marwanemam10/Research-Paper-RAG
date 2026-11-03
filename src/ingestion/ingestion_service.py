import os
import sys
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
    get_document_by_hash,
    upload_pdf_to_storage,
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

    def ingest_file(
        self,
        file_path: Union[str, Path],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        معالجة ملف PDF واحد من البداية حتى الفهرسة في Qdrant و Supabase.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        filename = path.name
        print(f"\n" + "-" * 50)
        print(f">> Processing: {filename}")
        print("-" * 50)

        # 1. حساب الـ Hash لمنع التكرار (Deduplication)
        file_hash = compute_file_hash(path)
        print(f"[Hash] SHA-256: {file_hash[:12]}...")

        # في حال كان Supabase غير مفعل/غير مهيأ
        if not self.supabase:
            print("[Warning] Supabase credentials not found. Proceeding with Qdrant indexing only.")
            chunks = split_pdf(path)
            for c in chunks:
                c.metadata["file_hash"] = file_hash
                c.metadata["source"] = filename
            index_documents(documents=chunks, collection_name=self.collection_name, client=self.qdrant)
            return {"status": "indexed_local", "filename": filename, "total_chunks": len(chunks)}

        # 2. التحقق من وجود المستند في جدول documents
        existing_doc = get_document_by_hash(self.supabase, file_hash)
        if existing_doc and existing_doc.get("status") == "indexed" and not force:
            print(f"[Deduplication] '{filename}' is already indexed (Doc ID: {existing_doc['id']}). Skipping.")
            return {
                "status": "already_indexed",
                "doc_id": existing_doc["id"],
                "filename": filename,
                "file_hash": file_hash,
                "total_chunks": existing_doc.get("total_chunks", 0)
            }

        # إذا طُلب force وكان المستند مفهرساً سابقاً، نحذف الـ Chunks القديمة من Qdrant
        if existing_doc and force:
            old_id = existing_doc["id"]
            print(f"[Reindex] Removing existing chunks for doc_id: {old_id} from Qdrant...")
            delete_documents_by_file_id(file_id=old_id, collection_name=self.collection_name, client=self.qdrant)

        doc_id = None
        try:
            # 3. رفع الـ PDF إلى Supabase Storage Bucket
            print(f"[Storage] Uploading raw PDF to Supabase Storage '{DEFAULT_BUCKET_NAME}/{file_hash}.pdf'...")
            storage_path = upload_pdf_to_storage(
                client=self.supabase,
                file_path=path,
                file_hash=file_hash,
                bucket_name=DEFAULT_BUCKET_NAME
            )

            # 4. تسجيل المستند بحالة 'processing' في جدول documents
            doc_record = create_or_update_document_record(
                client=self.supabase,
                filename=filename,
                file_hash=file_hash,
                storage_path=storage_path
            )
            doc_id = doc_record["id"]
            print(f"[DB] Document record initialized (ID: {doc_id}, status: 'processing').")

            # 5. التقطيع إلى Chunks
            print(f"[Chunking] Splitting PDF into chunks...")
            chunks = split_pdf(path)
            total_chunks = len(chunks)
            print(f"[Chunking] Generated {total_chunks} chunks.")

            # 6. حقن الـ file_id و file_hash و source في الـ metadata
            for chunk in chunks:
                chunk.metadata["file_id"] = str(doc_id)
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["source"] = filename

            # 7. التضمين والرفع إلى Qdrant
            print(f"[Qdrant] Indexing chunks into collection '{self.collection_name}'...")
            index_documents(
                documents=chunks,
                collection_name=self.collection_name,
                client=self.qdrant,
                recreate_collection=False
            )

            # 8. تحديث الحالة في Supabase إلى 'indexed'
            mark_document_indexed(client=self.supabase, doc_id=doc_id, total_chunks=total_chunks)
            print(f"[Success] Successfully ingested and indexed '{filename}' ({total_chunks} chunks).")

            return {
                "status": "indexed",
                "doc_id": doc_id,
                "filename": filename,
                "file_hash": file_hash,
                "total_chunks": total_chunks
            }

        except Exception as e:
            err_msg = str(e)
            print(f"[Error] Failed to ingest '{filename}': {err_msg}")
            if doc_id and self.supabase:
                try:
                    mark_document_failed(client=self.supabase, doc_id=doc_id, error_message=err_msg)
                except Exception as db_err:
                    print(f"[DB Warning] Could not update failed status: {db_err}")
            raise e

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
