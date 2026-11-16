"""
Figure Processor
----------------

Production Version (Full Extraction)

Processes cropped figures and preserves ALL figures containing meaningful text.
Ensures zero information loss from:
✓ Infographics
✓ Strategy & Demographics charts
✓ Product Collections & Catalogs
✓ Financial Charts & Tables
"""

import gc
from pathlib import Path

from ingestion.models.document_schema import Figure
from ingestion.vision.image_cropper import ImageCropper
from ingestion.analyzers.figure_analyzer import FigureAnalyzer


class FigureProcessor:

    def __init__(self):

        print("Figure Processor Ready.\n")
        self.cropper = ImageCropper()
        self.analyzer = FigureAnalyzer()

    # ----------------------------------------------------------

    def process(
        self,
        pdf_path,
        docling_document
    ):

        cropped_images = self.cropper.extract(
            pdf_path,
            docling_document
        )

        figures = []

        for image in cropped_images:

            try:

                analysis = self.analyzer.analyze(
                    image["path"]
                )

                if analysis is None:
                    continue

                ocr_text = analysis["ocr_text"].strip()
                figure_type = analysis["figure_type"]

                # -----------------------------------
                # Keep ANY figure with meaningful text
                # -----------------------------------
                if not ocr_text:
                    continue

                # Skip tiny noise / logos with less than 5 characters
                if len(ocr_text) < 5 and figure_type in {"logo", "icon"}:
                    continue

                figure = Figure(
                    image_path=analysis["image_path"],
                    caption="",
                    ocr_text=ocr_text,
                    figure_type=figure_type,
                    width=analysis["width"],
                    height=analysis["height"],
                    source="docling",
                    page_number=image["page"]
                )

                figures.append(figure)

            finally:
                gc.collect()

        print(f"Useful Figures Captured: {len(figures)} / {len(cropped_images)}")

        return figures
