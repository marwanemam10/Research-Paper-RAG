import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# ضبط الـ encoding لـ UTF-8 على أنظمة Windows
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from src.ingestion.ingestion_service import IngestionService
from src.storage.supabase_client import get_supabase_client

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "src" / "data"


def main():
    parser = argparse.ArgumentParser(
        description="Dedicated Ingestion Pipeline CLI: Upload, Deduplicate, and Index Academic Papers"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a single PDF research paper to ingest"
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing PDF files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if document hash already exists in Supabase"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="research_papers",
        help="Target Qdrant vector collection name (default: 'research_papers')"
    )

    args = parser.parse_args()

    # فحص إعدادات Supabase وتنبيه المستخدم إن لم تكن مهيأة
    supabase_client = get_supabase_client()
    if not supabase_client:
        print("\n" + "!" * 60)
        print("[WARNING] Supabase environment variables not configured!")
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env to enable:")
        print("  1. SHA-256 Deduplication via PostgreSQL")
        print("  2. PDF file storage in 'research-papers' bucket")
        print("Proceeding in Local Vector Store mode (Qdrant only).")
        print("!" * 60 + "\n")

    service = IngestionService(collection_name=args.collection)

    try:
        if args.file:
            target_file = Path(args.file)
            if not target_file.is_file():
                print(f"[Error] File not found: {target_file}")
                sys.exit(1)
            result = service.ingest_file(target_file, force=args.force)
            print("\n[Result Summary]:", result)
        else:
            target_dir = Path(args.dir)
            if not target_dir.is_dir():
                print(f"[Error] Directory not found: {target_dir}")
                sys.exit(1)
            results = service.ingest_directory(target_dir, force=args.force)
            print(f"\n[Batch Complete] Successfully processed {len(results)} file(s).")

    except Exception as e:
        print(f"\n[Fatal Error during ingestion]: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

