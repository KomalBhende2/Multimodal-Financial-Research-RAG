"""
PDF Loader
----------
Loads all PDFs and attaches document metadata.
Skips duplicate PDF paths.
"""

from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    def __init__(self):

        self.documents_folder = Path("data/documents")
        self.metadata_folder = Path("data/metadata")

    def load_documents(self):

        all_documents = []

        metadata_files = sorted(self.metadata_folder.glob("*.json"))

        print(f"\nFound {len(metadata_files)} metadata files.\n")

        loaded_pdfs = set()

        for metadata_file in metadata_files:

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            pdf_path = Path(metadata["filepath"]).resolve()

            # Skip duplicate PDFs
            if pdf_path in loaded_pdfs:
                print(f"Skipping duplicate PDF: {pdf_path.name}")
                continue

            loaded_pdfs.add(pdf_path)

            if not pdf_path.exists():
                print(f"PDF not found: {pdf_path}")
                continue

            print(f"Loading: {pdf_path.name}")

            loader = PyPDFLoader(str(pdf_path))

            pages = loader.load()

            for page in pages:
                page.metadata.update(metadata)

            all_documents.extend(pages)

        print(f"\nTotal Pages Loaded: {len(all_documents)}")

        return all_documents