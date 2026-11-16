import unittest
import pymupdf
from pathlib import Path
import re

from ingestion.processors.page_classifier import PageLayoutClassifier, PageType
from ingestion.processors.spatial_layout_engine import SpatialLayoutEngine
from ingestion.processors.page_validator import PageValidator
from ingestion.processors.document_builder import DocumentBuilder
from ingestion.extractors.table_extractor import TableExtractor
from ingestion.processors.deduplicator import Deduplicator
from ingestion.models.document_schema import Document


class TestUniversalPipeline(unittest.TestCase):

    def setUp(self):
        p = Path("data/documents/investor_presentation_Results_Presentation_2026-08-10.pdf")
        if not p.exists():
            p = Path("data/documents/investor_presentation.pdf")
        self.pdf_path = p

    def test_page_classifier(self):
        doc = pymupdf.open(self.pdf_path)
        classifier = PageLayoutClassifier()
        
        # Page 14 & 20 should be structured 2D canvases
        self.assertEqual(classifier.classify(doc[13]), PageType.STRUCTURED_2D_CANVAS)
        self.assertEqual(classifier.classify(doc[19]), PageType.STRUCTURED_2D_CANVAS)
        # Page 24 (image only) should be scanned_raster
        self.assertEqual(classifier.classify(doc[23]), PageType.SCANNED_RASTER)
        doc.close()

    def test_spatial_engine_targets(self):
        doc = pymupdf.open(self.pdf_path)
        engine = SpatialLayoutEngine()
        
        # Page 14: Financial Snapshot
        p14 = engine.process_page(doc[13], 14)
        self.assertEqual(len(p14.tables), 1)
        self.assertEqual(p14.tables[0].source, "spatial_snapshot")
        self.assertGreaterEqual(len(p14.tables[0].rows), 5)
        
        # Page 15: Time-series bar charts
        p15 = engine.process_page(doc[14], 15)
        self.assertEqual(len(p15.tables), 1)
        self.assertEqual(p15.tables[0].source, "spatial_timeseries")
        self.assertIn("Q1FY25", p15.tables[0].rows[0])
        
        # Page 17: Multi-column cards
        p17 = engine.process_page(doc[16], 17)
        self.assertEqual(len(p17.tables), 1)
        self.assertEqual(p17.tables[0].source, "spatial_cards")
        
        # Page 18: Category cards
        p18 = engine.process_page(doc[17], 18)
        h3_paras = [p.text for p in p18.paragraphs if p.text.startswith("### ")]
        self.assertGreaterEqual(len(h3_paras), 5)
        
        # Page 19: Workflow chevrons
        p19 = engine.process_page(doc[18], 19)
        stages = [p.text for p in p19.paragraphs if p.text.startswith("### ")]
        self.assertIn("### Design", stages)
        self.assertIn("### Procurement", stages)
        self.assertIn("### Quality Control", stages)
        self.assertIn("### Inventory Management", stages)
        
        # Page 20: 2D Grid Table
        p20 = engine.process_page(doc[19], 20)
        self.assertEqual(len(p20.tables), 1)
        self.assertEqual(p20.tables[0].source, "spatial_grid")
        self.assertEqual(len(p20.tables[0].rows), 6) # header + 5 dimensions
        self.assertEqual(len(p20.tables[0].rows[0]), 6) # Dimension + 5 channels
        
        doc.close()

    def test_page_validator_recall(self):
        doc = pymupdf.open(self.pdf_path)
        engine = SpatialLayoutEngine()
        validator = PageValidator()
        
        # All target presentation pages should achieve >= 90% recall
        for p_num in [14, 15, 17, 18, 19, 20]:
            page_obj = engine.process_page(doc[p_num - 1], p_num)
            page_obj, recall = validator.validate_page(doc[p_num - 1], page_obj)
            self.assertGreaterEqual(recall, 0.90, f"Page {p_num} recall {recall:.1%} fell below 90%")
            
        doc.close()

    def test_table_extractor_rejects_sliced_words(self):
        # Sliced words ending with hyphen across columns
        cell_val = "creating trend-"
        has_sliced = bool(re.search(r"\b[a-zA-Z]{3,}-\s*$", cell_val))
        self.assertTrue(has_sliced)

    def test_synthetic_ocr_detection(self):
        apollo_pdf = Path("data/documents/investor_presentation_Announcement_under_Regulation_30_LODR_Investor_Presentation_2026-08-12.pdf")
        if apollo_pdf.exists():
            doc = pymupdf.open(apollo_pdf)
            classifier = PageLayoutClassifier()
            # Synthetic OCR pages should be classified as SCANNED_RASTER
            self.assertEqual(classifier.classify(doc[0]), PageType.SCANNED_RASTER)
            self.assertEqual(classifier.classify(doc[4]), PageType.SCANNED_RASTER)
            doc.close()


if __name__ == "__main__":
    unittest.main()
