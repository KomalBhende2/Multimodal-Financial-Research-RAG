"""
Production PDFPlumber Table Extractor
-------------------------------------

Purpose
-------
Extract tables that Docling may miss.

Returns:
{
    page_number: [
        Table(...),
        Table(...)
    ]
}
"""

from pathlib import Path
import pdfplumber

from ingestion.models.document_schema import Table


class PDFPlumberTableExtractor:

    def __init__(self):

        self.settings = {

            "vertical_strategy": "lines",

            "horizontal_strategy": "lines",

            "intersection_tolerance": 5,

            "snap_tolerance": 3,

            "join_tolerance": 3,

            "edge_min_length": 3

        }

    def extract(self, pdf_path):

        pdf_path = Path(pdf_path)

        extracted_tables = {}

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                page_tables = []

                tables = page.extract_tables(self.settings)

                if not tables:

                    continue

                for table in tables:

                    rows = []

                    for row in table:

                        cleaned = []

                        for cell in row:

                            if cell is None:
                                cleaned.append("")
                            else:
                                cleaned.append(
                                    str(cell).strip()
                                )

                        if any(cleaned):
                            rows.append(cleaned)

                    if len(rows) == 0:
                        continue

                    page_tables.append(

                        Table(

                            rows=rows,

                            source="pdfplumber",

                            confidence=0.95,

                            page_number=page_number

                        )

                    )

                if page_tables:

                    extracted_tables[page_number] = page_tables

        return extracted_tables