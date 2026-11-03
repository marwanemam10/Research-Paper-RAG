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

from src.retrieval.retriever import get_retriever, retrieve_documents
from src.generation.generation import generate_answer
from src.storage.vector_store import DEFAULT_COLLECTION_NAME, get_qdrant_client


def execute_query(
    question: str,
    retriever,
    session_id: str = "query_session",
    show_sources: bool = True
) -> str:
    """
    تنفيذ استعلام سريع ومباشر على Qdrant وتوليد الإجابة مع تسجيل Langfuse trace.
    """
    print(f"\n[Question]: {question}\n")

    # 1. استرجاع المستندات ذات الصلة مباشرة من Qdrant
    retrieved_docs = retrieve_documents(query=question, retriever=retriever, session_id=session_id)

    # 2. توليد الإجابة عبر Gemini
    answer = generate_answer(question=question, retriever=retriever, session_id=session_id)

    # عرض الإجابة
    print("=" * 20 + " Answer " + "=" * 20)
    print(answer)
    print("=" * 48)

    # عرض المصادر المستخدمة
    if show_sources and retrieved_docs:
        print("\n[Sources Used]:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            src_path = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", None)
            file_id = doc.metadata.get("file_id", None)
            src_name = Path(src_path).name if src_path else "Unknown"
            page_str = f" (Page {page_num + 1})" if page_num is not None else ""
            id_str = f" [Doc ID: {file_id[:8]}...]" if file_id else ""
            print(f"  [{idx}] {src_name}{page_str}{id_str}")
        print()

    return answer


def interactive_mode(retriever):
    """
    واجهة محادثة تفاعلية سريعة عبر سطر الأوامر (CLI).
    """
    print("\n--- Interactive Research Query Mode (Type 'exit', 'quit', or 'q' to stop) ---")
    session_id = "interactive_research_session"

    while True:
        try:
            question = input("\n[Ask a question]: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                print(">> Goodbye!")
                break

            execute_query(question=question, retriever=retriever, session_id=session_id)
        except (KeyboardInterrupt, EOFError):
            print("\n>> Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Dedicated Retrieval Pipeline CLI: Query Research Papers Vector Store"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single research question to ask"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive conversation loop"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of retrieved chunks for context (default: 4)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Target Qdrant collection name (default: '{DEFAULT_COLLECTION_NAME}')"
    )

    args = parser.parse_args()

    # فحص الاتصال بقاعدة بيانات Qdrant
    client = get_qdrant_client()
    try:
        if not client.collection_exists(args.collection):
            print(f"\n[ERROR] Collection '{args.collection}' does not exist in Qdrant.")
            print("Please run the Ingestion Pipeline first:")
            print("   python ingest.py\n")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Qdrant at {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
        print("Make sure Docker and Qdrant container are running:")
        print("   cd qdrant-local && docker compose up -d\n")
        sys.exit(1)

    # تهيئة الـ Retriever مباشرة دون أي فحص للملفات أو تقسيم
    retriever = get_retriever(collection_name=args.collection, k=args.k)

    if args.question:
        execute_query(question=args.question, retriever=retriever)
    elif args.interactive:
        interactive_mode(retriever)
    else:
        # إذا لم يحدد المستخدم سؤالاً أو وضعاً تفاعلياً، تشغيل سؤال توضيحي ثم الدخول على الوضع التفاعلي
        sample_q = "What is the key advantage of the Transformer architecture over recurrent models?"
        print(">> No question provided. Running a sample query:\n")
        execute_query(question=sample_q, retriever=retriever)
        interactive_mode(retriever)


if __name__ == "__main__":
    main()

