"""
Table Extractor
---------------

Production Version

Improvements:
✓ Better cleaning
✓ Detects table title
✓ Removes duplicate rows
✓ Removes fake/phantom single-column tables
✓ Rejects sliced diagram chevrons (words chopped across columns)
✓ Keeps page number
✓ Better structure
"""

from pathlib import Path
import re

import pdfplumber
from rapidfuzz import fuzz

from ingestion.models.document_schema import Table


class TableExtractor:

    def extract_tables(self, pdf_path):

        print(f"\nScanning tables : {Path(pdf_path).name}")

        page_tables = {}

        with pdfplumber.open(pdf_path) as pdf:

            for page_no, page in enumerate(pdf.pages, start=1):

                recovered = []
                seen_tables = []

                try:

                    tables = page.extract_tables()

                    if not tables:
                        page_tables[page_no] = []
                        continue

                    page_text = page.extract_text() or ""

                    page_lines = [
                        line.strip()
                        for line in page_text.split("\n")
                        if line.strip()
                    ]

                    for table in tables:

                        if not table:
                            continue

                        rows = []
                        seen_rows = set()

                        # -------------------------------------
                        # Clean rows
                        # -------------------------------------
                        for row in table:

                            if not row:
                                continue

                            cleaned = []

                            for cell in row:

                                if cell is None:
                                    cleaned.append("")
                                else:
                                    value = re.sub(
                                        r"\s+",
                                        " ",
                                        str(cell)
                                    ).strip()
                                    cleaned.append(value)

                            if not any(cleaned):
                                continue

                            key = tuple(cleaned)

                            if key in seen_rows:
                                continue

                            seen_rows.add(key)
                            rows.append(cleaned)

                        if len(rows) < 2:
                            continue

                        cols = max(len(r) for r in rows)

                        # Must have at least 2 columns
                        if cols < 2:
                            continue

                        # Reject false-positive diagram slicing (words split across columns by chevron lines)
                        has_sliced_words = False
                        for r in rows:
                            for c in r:
                                if re.search(r"\b[a-zA-Z]{3,}-\s*$", str(c)):
                                    has_sliced_words = True
                                    break
                            if has_sliced_words:
                                break
                        if has_sliced_words:
                            continue

                        table_text = "\n".join(
                            " | ".join(r)
                            for r in rows
                        )

                        if len(table_text) < 25:
                            continue

                        duplicate = False

                        for old in seen_tables:
                            if fuzz.ratio(old, table_text) > 95:
                                duplicate = True
                                break

                        if duplicate:
                            continue

                        seen_tables.append(table_text)

                        # -------------------------------------
                        # Guess table title
                        # -------------------------------------
                        title = "Recovered Table"
                        header = " ".join(rows[0]).strip()

                        if len(header) > 5:
                            title = header
                        else:
                            for line in page_lines:
                                if (
                                    len(line) > 5
                                    and len(line) < 100
                                    and "%" not in line
                                ):
                                    title = line
                                    break

                        recovered.append(
                            Table(
                                title=title,
                                rows=rows,
                                source="pdfplumber",
                                page_number=page_no
                            )
                        )

                except Exception as e:
                    print(f"Page {page_no}: {e}")

                page_tables[page_no] = recovered

        return page_tables
