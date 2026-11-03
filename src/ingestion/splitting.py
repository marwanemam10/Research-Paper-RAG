from pathlib import Path
from typing import Optional, List, Union
# pyrefly: ignore[missing-import]
from langchain_core.documents import Document
# pyrefly: ignore[missing-import]
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
# pyrefly: ignore[missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def split_pdf(
    file_path: Union[str, Path],
    chunk_size: int = 750,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    قراءة ملف PDF واحد وتقطيعه إلى Chunks مع الحفاظ على الـ Metadata ورقم الصفحة.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(docs)
    return splits


def get_document_splits(
    data_dir: Optional[Union[str, Path]] = None, 
    chunk_size: int = 750, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    تحميل جميع ملفات الـ PDF من مجلد وتقطيعها إلى Chunks.
    """
    if data_dir is None:
        target_path = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path("data")
    else:
        target_path = Path(data_dir)

    if not target_path.exists():
        raise FileNotFoundError(f"Data directory not found: {target_path}")

    loader = PyPDFDirectoryLoader(str(target_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(docs)
    return splits
