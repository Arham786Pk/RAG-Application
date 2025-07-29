import os
from typing import List
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content)]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()
