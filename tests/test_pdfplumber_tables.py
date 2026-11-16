from pathlib import Path

from ingestion.extractors.pdfplumber_table_extractor import (
    PDFPlumberTableExtractor
)

pdf = Path("data/documents/Ather AR.pdf")

extractor = PDFPlumberTableExtractor()

tables = extractor.extract(pdf)

print()

print("=" * 80)
print("TABLE SUMMARY")
print("=" * 80)

print()

print("Pages containing tables:", len(tables))

print()

for page_no, page_tables in tables.items():

    print(f"Page {page_no}")

    print(f"Tables: {len(page_tables)}")

    print()

    first = page_tables[0]

    print("Rows:", len(first.rows))

    print()

    for row in first.rows[:5]:

        print(row)

    print("-" * 60)