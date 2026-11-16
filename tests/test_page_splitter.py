from pathlib import Path

from ingestion.extractors.page_splitter import PDFPageSplitter

pdf = Path("data/documents") / "Q1FY26 PPT.pdf"

splitter = PDFPageSplitter()

pages = splitter.split_pdf(pdf)

print()

print("Total Pages:", len(pages))

print()

for p in pages:
    print(p)