class Retriever:

    def __init__(self, vector_store, top_k=5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query):

        documents = self.vector_store.similarity_search(
            query,
            k=self.top_k
        )

        return documents