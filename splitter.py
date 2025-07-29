from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding & retrieval.

    Args:
        documents (List[Document]): The documents to split.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks for context retention.

    Returns:
        List[Document]: List of split document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    return splitter.split_documents(documents)
