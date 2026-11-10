"""Vector store module for document embedding and retrieval"""

from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStore:
    """Manages vector store operations"""

    def __init__(self):
        """Initialize local Hugging Face embeddings"""

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
        )

        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""

        self.vectorstore = FAISS.from_documents(documents, self.embedding)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        """Get retriever instance"""

        if self.retriever is None:
            raise ValueError(
                "Vector store not initialized. Call create_vectorstore first."
            )

        return self.retriever

    def retrieve(self, query: str, k: int = 4):
        """Retrieve relevant documents"""

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(query, k=k)
