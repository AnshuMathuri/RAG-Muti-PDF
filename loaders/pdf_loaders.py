from typing import List
from langchain_community.document_loaders import PyPDFLoader

class DocumentService:
    """
        Responsible for loading PDF documents.
    """
    def __init__(self):
        self.documents=[]


    def load_pdf(self,file_path):
        """
         Load single PDF
        """
        loader=PyPDFLoader(file_path)
        pdf_documents=loader.load()
        return pdf_documents

    def load_pdfs(self,file_paths):

        """
        Load multiple PDF files.

        Parameters:
            file_paths: list of PDF paths

        Returns:
            Combined list of documents
        """
        self.documents=[]

        for file_path in file_paths:

            print(
                f"Loading PDF :{file_path}"
                )
            documents=self.load_pdf(file_path)
            self.documents.extend(documents)
            
        return self.documents

    def get_document_count(self):
        """
        Return total loaded pages.
        """

        return len(self.documents)