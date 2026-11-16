from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from docling_core.types.doc import PictureItem

# Suppress false-alarm OCR warnings from Docling when inspecting blank or graphic crops
logging.getLogger("docling.models.stages.ocr.rapid_ocr_model").setLevel(logging.ERROR)
logging.getLogger("docling").setLevel(logging.WARNING)


class DoclingExtractor:

    def __init__(self):

        print("Loading Docling...")
        
        try:
            # Docling 2.x: OCR is enabled automatically if available
            self.converter = DocumentConverter()
            print("[OK] Docling initialized (OCR will be used if available)")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Docling: {e}")
            raise

        print("Docling Ready.\n")

    def extract(self, pdf_path):

        pdf_path = Path(pdf_path)
        
        print(f"\n[EXTRACTING] Processing: {pdf_path.name}")

        result = self.converter.convert(str(pdf_path))
        
        doc = result.document
        print(f"  Total pages: {doc.num_pages()}")

        # Return the complete DoclingDocument
        return doc
