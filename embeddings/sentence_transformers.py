"""
sentence_transformer.py
------------------------
Wraps a sentence-transformers model to turn text chunks into
dense vector embeddings for storage in / retrieval from Pinecone.
"""

from typing import List
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:

    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)

    def embed_documents(self,texts:List[str]) ->List[List[float]]:
        """Generate enbeddings for multiple texts."""
        embeddings=self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self,text:str) -> List[float]:
        """Generate embedding for one query."""

        embeddings=self.model.encode(text)
        return embeddings.tolist()
    
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()