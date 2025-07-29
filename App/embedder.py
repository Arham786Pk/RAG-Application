import os
os.environ["TRANSFORMERS_NO_TF"] = "1" # Disable TensorFlow to avoid unnecessary dependencies

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a HuggingFaceEmbeddings object with the specified model.
    
    You can replace the model with any supported sentence-transformer model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
