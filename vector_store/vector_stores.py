from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

class VectorStoreService:

    def __init__(
        self,
        pinecone_api_key,
        embedder,
        index_name="multi-query-rag-index"
    ):

        self.index_name = index_name
        self.embedder = embedder

        self.pc = Pinecone(
            api_key=pinecone_api_key
        )

        self.create_index()

        self.index = self.pc.Index(
            self.index_name
        )

    def create_index(self):
        existing_indexes=[index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(name=self.index_name,
                                 dimension=384,
                                 metric="cosine",
                                 spec=ServerlessSpec(cloud="aws",
                                                     region="us-east-1"))

    def store_documents(self,docs):
        VectorStore=(
            PineconeVectorStore.from_documents(docs,self.embedder,
                                               index_name=self.index_name)
        )
        return VectorStore