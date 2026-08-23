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

    def retrieve(self, query):

        # Query ka embedding
        query_vector = self.embedder.embed_query(query)

        # Pinecone semantic search
        results = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])

        return matches