from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document


class RAGState(BaseModel):
    """State object for RAG workflow"""

    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
