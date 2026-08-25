import tempfile

from config import (
    GROQ_API_KEY,
    PINECONE_API_KEY
)

from loaders.pdf_loaders import DocumentService
from splitters.chunking import TextSplitters
from embeddings.sentence_transformers import SentenceTransformerEmbedder
from vector_store.vector_stores import VectorStoreService
from retriever.retriever import Retriever
from llm.llm_clients import LLMClient


class RAGPipeline:

    def __init__(self):

        # 1. Document Loader
        self.loader = DocumentService()

        # 2. Text Splitter
        self.splitter = TextSplitters(
            chunk_size=1000,
            chunk_overlap=100
        )

        # 3. Embedding Model
        self.embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2"
        )

        # 4. Vector Store
        self.vector_store = VectorStoreService(
            pinecone_api_key=PINECONE_API_KEY,
            embedder=self.embedder
        )

        # 5. Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store.get_vector_store(),
            top_k=5
        )

        # 6. LLM
        self.llm = LLMClient(
            api_key=GROQ_API_KEY
        )

    # =====================================================
    # PROCESS DOCUMENT
    # =====================================================

    def process_document(self, uploaded_file):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name

        # Load PDF
        documents = self.loader.load_pdf(
            pdf_path
        )

        # Split into chunks
        chunks = self.splitter.split_documents(
            documents
        )

        # Store chunks in Pinecone
        self.vector_store.store_documents(
            chunks
        )

        return len(chunks)

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(self, question):

        # Retrieve relevant documents
        results = self.retriever.retrieve(
            question
        )

        # Extract text from LangChain Documents
        context = "\n\n".join(
            document.page_content
            for document in results
        )

        # Generate answer
        return self.llm.generate(
            context=context,
            question=question
        )