import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# pyrefly: ignore[missing-import]
from langfuse.langchain import CallbackHandler
from src.models.models import model
# pyrefly: ignore[missing-import]
from src.retrieval.retreiver import get_retriever
from src.prompts.orchestrator import PromptOrchestrator

load_dotenv()


def format_docs(docs) -> str:
    """
    دمج محتوى المستندات المسترجعة مع توثيق المصدر ورقم الصفحة إن وجد.
    """
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)
        header = f"[Doc {i} | Source: {source}" + (f" | Page: {page + 1}]" if page is not None else "]")
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever=None, prompt_version: str = "v1"):
    if retriever is None:
        retriever = get_retriever(k=4)
    prompt = PromptOrchestrator.get_prompt("rag", version=prompt_version)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


def generate_answer(
    question: str,
    retriever=None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    توليد الإجابة مع تتبع كامل لخطوات الـ RAG عبر Langfuse.
    """
    chain = build_rag_chain(retriever=retriever)

 
    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "session_id": session_id or "default_session",
            "user_id": user_id or "default_user",
            **(metadata or {}),
        },
    }

    
    response = chain.invoke(question, config=config)
    return response