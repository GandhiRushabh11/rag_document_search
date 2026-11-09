"""Document processing module for loading and spliting documents"""

from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)


class DocumentProcessor:
    """Handle document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.spiltter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load Documents from urls"""

        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load Documents from all PDFs inside a directory"""

        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load Document(s) from TXT file"""

        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load Document(s) from PDF file"""

        loader = PyPDFLoader(str("data"))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:

        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self, self.load_from_url(src))

            path = Path("data")

            if path.is_dir():
                docs.extend(self, self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self, self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type:{src}."
                    "Use URL, .txt file or PDF directory."
                )
            return docs

    def spilt_documents(self, documents: List[Document]) -> List[Document]:
        return self.spiltter.split_documents(documents)

    def process_url(self, urls: List[str]) -> List[Document]:
        docs = self.load_documents(urls)
        return self.spilt_documents(docs)
