from pathlib import Path
from typing import Optional
# pyrefly: ignore[missing-import]
from langchain_community.document_loaders import PyPDFDirectoryLoader
# pyrefly: ignore[missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

# المسار الافتراضي لمجلد البيانات داخل المشروع: src/data
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_document_splits(
    data_dir: Optional[str] = None, 
    chunk_size: int = 750, 
    chunk_overlap: int = 100
):
    """
    تحميل ملفات الـ PDF وتقطيعها إلى Chunks جاهزة للـ Indexing.
    """
    if data_dir is None:
        target_path = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path("data")
    else:
        target_path = Path(data_dir)

    if not target_path.exists():
        raise FileNotFoundError(f"مجلد البيانات غير موجود: {target_path}")

    loader = PyPDFDirectoryLoader(str(target_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(docs)
    return splits
