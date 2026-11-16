"""
Run Complete Ingestion for all PDFs in data/documents
"""

from pathlib import Path
from ingestion.pipeline import IngestionPipeline


INPUT = Path("data/documents")
OUTPUT = Path("ingestion/output")

OUTPUT.mkdir(exist_ok=True, parents=True)

pipeline = IngestionPipeline()
pdfs = list(INPUT.glob("*.pdf"))

print(f"\nFound {len(pdfs)} PDFs\n")

for pdf in pdfs:
    document = pipeline.process_pdf(pdf)
    output_file = OUTPUT / f"{pdf.stem}.md"
    print(f"Saved -> {output_file}")

print("\nDone")
