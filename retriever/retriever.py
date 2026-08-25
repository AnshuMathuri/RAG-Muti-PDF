from typing import List
from langchain_core.documents import Document


class Retriever:

    def __init__(
        self,
        index,
        embedder,
        top_k=5
    ):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:

        # Convert query into embedding
        query_vector = self.embedder.embed_query(query)

        # Query Pinecone
        response = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        documents = []

        for match in response.matches:

            # Safely get metadata
            metadata = getattr(match, "metadata", None)

            if not metadata:
                continue

            # Pinecone metadata key created by LangChain
            text = metadata.get("text")

            # Some versions/configurations may use page_content
            if not text:
                text = metadata.get("page_content")

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        return documents