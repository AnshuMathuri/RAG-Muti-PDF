from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitters:

    def __init__(self,chunk_size:int =1000 , chunk_overlap :int =100):


        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)

    def split_documents(self,documents):
        """
        Split documents into chunks.
        """

        if not documents:
            return []
        chunks= self.splitter.split_documents(documents)

        return chunks

    def chunk_size_count(self,chunks):
        """
        Return number of chunks
        """
        return len(chunks)