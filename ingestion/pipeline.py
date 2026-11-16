"""
Memory Optimized Ingestion Pipeline
----------------------------------

Processes PDFs in batches (default: 5 pages).

This prevents Docling, OCR and Table extraction
from loading the entire PDF into RAM.
"""

from pathlib import Path
import gc
import pymupdf as fitz

from ingestion.extractors.docling_extractor import DoclingExtractor
from ingestion.extractors.table_extractor import TableExtractor

from ingestion.processors.document_parser import DocumentParser
from ingestion.processors.metadata_enricher import MetadataEnricher
from ingestion.processors.figure_processor import FigureProcessor
from ingestion.processors.text_cleaner import TextCleaner
from ingestion.processors.deduplicator import Deduplicator
from ingestion.processors.document_builder import DocumentBuilder
from ingestion.processors.spatial_layout_engine import SpatialLayoutEngine
from ingestion.processors.page_classifier import PageLayoutClassifier, PageType
from ingestion.processors.page_validator import PageValidator
from ingestion.models.document_schema import Paragraph


class IngestionPipeline:

    def __init__(self, batch_size=5):

        self.batch_size = batch_size

        self.docling = DoclingExtractor()
        self.parser = DocumentParser()
        self.metadata = MetadataEnricher()
        self.tables = TableExtractor()
        self.figures = FigureProcessor()
        self.cleaner = TextCleaner()
        self.deduplicator = Deduplicator()
        self.builder = DocumentBuilder()
        self.spatial = SpatialLayoutEngine()
        self.classifier = PageLayoutClassifier()
        self.validator = PageValidator()
        self.ocr = self.figures.analyzer.ocr

    # ----------------------------------------------------
    # Create Batch PDF
    # ----------------------------------------------------

    def _save_batch(self, pdf_path, start_page, end_page):

        original = fitz.open(pdf_path)
        batch_pdf = fitz.open()

        for page in range(start_page, end_page):
            batch_pdf.insert_pdf(
                original,
                from_page=page,
                to_page=page
            )

        batch_folder = Path("temp_pdf_batches")
        batch_folder.mkdir(exist_ok=True)

        temp_pdf = batch_folder / f"batch_{start_page+1}_{end_page}.pdf"

        if temp_pdf.exists():
            temp_pdf.unlink()

        batch_pdf.save(str(temp_pdf))
        batch_pdf.close()
        original.close()

        return temp_pdf

    # ----------------------------------------------------
    # MAIN
    # ----------------------------------------------------

    def process_pdf(self, pdf_path):

        pdf_path = Path(pdf_path)

        pdf = fitz.open(pdf_path)
        total_pages = len(pdf)
        pdf.close()

        print("=" * 70)
        print("Processing :", pdf_path.name)
        print("Total Pages :", total_pages)
        print("Batch Size :", self.batch_size)
        print("=" * 70)

        all_pages = []
        recovered_tables = {}
        recovered_figures = []
        batch_folder = Path("temp_pdf_batches")

        # ------------------------------------------------

        for start in range(0, total_pages, self.batch_size):

            end = min(start + self.batch_size, total_pages)

            print(f"\nProcessing Pages {start+1} -> {end}")

            temp_pdf = self._save_batch(
                pdf_path,
                start,
                end
            )

            try:

                # ------------------------------
                # Docling
                # ------------------------------
                docling_doc = self.docling.extract(temp_pdf)

                # ------------------------------
                # Parse
                # ------------------------------
                document = self.parser.parse(
                    docling_doc,
                    pdf_path.name
                )

                # ------------------------------
                # Tables
                # ------------------------------
                batch_tables = self.tables.extract_tables(temp_pdf)

                # ------------------------------
                # Layout Classification & Spatial Enhancement
                # ------------------------------
                pdf_doc = fitz.open(temp_pdf)
                for page_idx in range(len(pdf_doc)):
                    if page_idx < len(document.pages):
                        page_type = self.classifier.classify(pdf_doc[page_idx])
                        if page_type == PageType.SCANNED_RASTER:
                            # Do NOT apply vector spatial coordinate projections on raster/synthetic-OCR pages
                            continue

                        spatial_page = self.spatial.process_page(pdf_doc[page_idx], page_idx + 1)
                        has_spatial_table = any(t.source.startswith("spatial_") for t in spatial_page.tables)
                        has_card_sections = any(p.heading_level == 3 for p in spatial_page.paragraphs)
                        if page_type == PageType.STRUCTURED_2D_CANVAS or has_spatial_table or has_card_sections:
                            document.pages[page_idx] = spatial_page
                            if has_spatial_table:
                                batch_tables[page_idx + 1] = []

                for page_no, tables in batch_tables.items():
                    original_page = page_no + start
                    recovered_tables[original_page] = tables

                # ------------------------------
                # Figures (Keep ALL with text)
                # ------------------------------
                batch_figures = self.figures.process(
                    temp_pdf,
                    docling_doc
                )

                for fig in batch_figures:
                    fig.page_number += start

                recovered_figures.extend(batch_figures)

                # ------------------------------
                # Fallback OCR for sparse/visual slides
                # ------------------------------
                # If a slide has fewer than 25 words of text and no tables (e.g. Infographics,
                # Demographics, Strategy cards, Schemes), OCR the full slide to guarantee zero text loss.
                for page_idx, page in enumerate(document.pages):
                    page_no_abs = page_idx + start + 1
                    page_tables_count = len(batch_tables.get(page_idx + 1, [])) + len(page.tables)
                    total_words = sum(len(p.text.split()) for p in page.paragraphs)

                    # Trigger full-slide OCR if content is sparse
                    if total_words < 25 and page_tables_count == 0:
                        try:
                            pdf_page = pdf_doc.load_page(page_idx)
                            pix = pdf_page.get_pixmap(dpi=150)
                            temp_page_img = batch_folder / f"temp_page_ocr_{page_idx}.png"
                            pix.save(str(temp_page_img))
                            ocr_text = self.ocr.extract_text(temp_page_img)
                            if temp_page_img.exists():
                                temp_page_img.unlink()

                            if ocr_text:
                                seen_lines = {p.text.strip().lower() for p in page.paragraphs}
                                for line in ocr_text.splitlines():
                                    line_clean = line.strip()
                                    if line_clean and line_clean.lower() not in seen_lines:
                                        seen_lines.add(line_clean.lower())
                                        page.paragraphs.append(
                                            Paragraph(text=line_clean, heading_level=0, source="full_page_ocr")
                                        )
                                if not page.title and page.paragraphs:
                                    page.title = page.paragraphs[0].text
                        except Exception:
                            pass
                pdf_doc.close()

                # Fix page numbers
                for page in document.pages:
                    page.page_number += start

                all_pages.extend(document.pages)

            finally:

                # Delete temporary PDF
                try:
                    temp_pdf.unlink()
                except Exception:
                    pass

                gc.collect()

        # ------------------------------------------------

        document.pages = all_pages

        print()
        print("Pages Parsed :", len(document.pages))

        document = self.metadata.enrich(document)

        for fig in recovered_figures:
            if 1 <= fig.page_number <= len(document.pages):
                document.pages[
                    fig.page_number - 1
                ].figures.append(fig)

        document = self.cleaner.clean(document)

        document = self.deduplicator.merge(
            document,
            recovered_tables
        )

        # ----------------------------------------------------
        # Audit Completeness with PageValidator
        # ----------------------------------------------------
        orig_doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(document.pages):
            if page_idx < len(orig_doc):
                document.pages[page_idx], recall = self.validator.validate_page(
                    orig_doc[page_idx], page
                )
        orig_doc.close()

        output_file = self.builder.build(
            document,
            output_folder="ingestion/output"
        )

        return document
