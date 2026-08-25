from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from config import PINECONE_INDEX_NAME


class VectorStoreService:

    def __init__(
        self,
        pinecone_api_key,
        embedder,
        index_name=PINECONE_INDEX_NAME
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

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embedder
        )

    def create_index(self):

        existing_indexes = self.pc.list_indexes()

        index_names = [
            index.name
            for index in existing_indexes
        ]

        if self.index_name not in index_names:

            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    def store_documents(self, docs):

        self.vector_store.add_documents(
            docs
        )

    def get_retriever(self, top_k=5):

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k
            }
        )