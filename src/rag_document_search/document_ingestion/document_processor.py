"""Document processing module for loading and splitting documents."""

from typing import List, Union
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)


class DocumentProcessor:
    """Handle document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load all PDFs from a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a TXT file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load documents from URLs, TXT files, or PDFs."""

        docs: List[Document] = []

        for src in sources:
            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))

            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))

            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))

            else:
                raise ValueError(f"Unsupported source type: {src}")

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)

    def process_urls(self, urls: List[str]) -> List[Document]:
        """Load and split documents from URLs."""
        docs = self.load_documents(urls)
        return self.split_documents(docs)
