import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# ضبط الـ encoding لـ UTF-8 على أنظمة Windows لتجنب مشاكل الحروف والـ terminal
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from src.ingestion.splitting import get_document_splits
from src.ingestion.indexing import index_documents, get_qdrant_client, get_vector_store
from src.retrieval.retreiver import get_retriever, retrieve_documents
from src.generation.generation import generate_answer


COLLECTION_NAME = "research_papers"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "src" / "data"


def initialize_rag_pipeline(
    collection_name: str = COLLECTION_NAME,
    data_dir: Path = DEFAULT_DATA_DIR,
    force_reindex: bool = False,
    k: int = 4
):
    """
    تهيئة الـ Pipeline والتأكد من وجود البيانات وفهرستها داخل Qdrant.
    إذا كانت البيانات مفهرسة مسبقاً، يتم تخطي الـ Chunking والـ Embedding بالكامل للبدء الفوري.
    """
    print("=" * 60)
    print(">> Initializing Academic Research RAG Pipeline...")
    print("=" * 60)

    client = get_qdrant_client()

    # التحقق من تشغيل سيرفر Qdrant
    try:
        collection_exists = client.collection_exists(collection_name)
    except Exception as e:
        print(f"\n[ERROR] Unable to connect to Qdrant at {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
        print("Please ensure Docker Desktop is running and start your container:")
        print("   cd qdrant-local && docker compose up -d\n")
        sys.exit(1)

    need_indexing = force_reindex

    # 1. فحص وجود الـ Collection وعدد الـ points المحفوظة مسبقاً
    if not force_reindex and collection_exists:
        points_count = client.count(collection_name).count
        if points_count > 0:
            print(f"[Success] Found indexed collection '{collection_name}' with {points_count} vectors.")
            print("[Info] Skipping PDF loading, Chunking, and Embedding (instant start).")
            vector_store = get_vector_store(collection_name=collection_name, client=client)
            retriever = get_retriever(vector_store=vector_store, collection_name=collection_name, k=k)
            print(">> Pipeline is ready to answer questions!")
            print("=" * 60 + "\n")
            return retriever
        else:
            print(f"[Warning] Collection '{collection_name}' exists but is empty.")
            need_indexing = True
    else:
        if force_reindex:
            print(f"[Info] Force re-index requested.")
        else:
            print(f"[Info] Collection '{collection_name}' not found.")
        need_indexing = True

    # 2. في حال كانت البيانات غير مفهرسة فقط: يتم تحميل الـ PDFs، التقطيع، والتضمين
    print(f"\n[Step 1] Loading & Splitting PDFs from: '{data_dir}'...")
    splits = get_document_splits(data_dir=str(data_dir))
    total_chunks = len(splits)
    print(f"[Success] Extracted {total_chunks} chunks.")

    print(f"\n[Step 2] Generating Embeddings & Indexing into Qdrant ('{collection_name}')...")
    vector_store = index_documents(
        documents=splits,
        collection_name=collection_name,
        client=client,
        recreate_collection=force_reindex,
        batch_size=50
    )

    # إنشاء كائن الـ Retriever
    retriever = get_retriever(vector_store=vector_store, collection_name=collection_name, k=k)
    print(">> Pipeline is ready to answer questions!")
    print("=" * 60 + "\n")
    return retriever


def query_pipeline(
    question: str,
    retriever,
    session_id: str = "rag_session",
    show_sources: bool = True
) -> str:
    """
    تنفيذ استعلام كامل: استرجاع المصادر، توليد الإجابة، وتسجيل الـ Trace في Langfuse.
    """
    print(f"\n[Question]: {question}\n")

    # 1. استرجاع المستندات ذات الصلة
    retrieved_docs = retrieve_documents(query=question, retriever=retriever, session_id=session_id)

    # 2. توليد الإجابة عبر نموذج Gemini
    answer = generate_answer(question=question, retriever=retriever, session_id=session_id)

    # عرض الإجابة
    print("=" * 20 + " Answer " + "=" * 20)
    print(answer)
    print("=" * 48)

    # عرض المصادر المسترجعة
    if show_sources and retrieved_docs:
        print("\n[Sources Used]:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            src_path = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", None)
            src_name = Path(src_path).name if src_path else "Unknown"
            page_str = f" (Page {page_num + 1})" if page_num is not None else ""
            print(f"  [{idx}] {src_name}{page_str}")
        print()

    return answer


def interactive_mode(retriever):
    """
    واجهة تفاعلية للأسئلة عبر سطر الأوامر (CLI)
    """
    print("\n--- Entering Interactive Mode (Type 'exit', 'quit', or 'q' to stop) ---")
    session_id = "interactive_cli_session"

    while True:
        try:
            question = input("\n[Ask a question]: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                print(">> Exiting RAG Pipeline. Goodbye!")
                break

            query_pipeline(question=question, retriever=retriever, session_id=session_id)
        except (KeyboardInterrupt, EOFError):
            print("\n>> Exiting RAG Pipeline. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="End-to-End Academic Research RAG Pipeline")
    parser.add_argument("-q", "--question", type=str, help="Single question to query the RAG pipeline with")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of all PDFs in data directory")
    parser.add_argument("--k", type=int, default=4, help="Number of context chunks to retrieve (default: 4)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive questioning mode")
    args = parser.parse_args()

    retriever = initialize_rag_pipeline(
        collection_name=COLLECTION_NAME,
        data_dir=DEFAULT_DATA_DIR,
        force_reindex=args.reindex,
        k=args.k
    )

    if args.question:
        query_pipeline(question=args.question, retriever=retriever)
    elif args.interactive:
        interactive_mode(retriever)
    else:
        sample_q = "What is the key advantage of the Transformer architecture over recurrent models?"
        print(">> No specific question passed. Running a sample test query:\n")
        query_pipeline(question=sample_q, retriever=retriever)
        
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
