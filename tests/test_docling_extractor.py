from pathlib import Path

from ingestion.extractors.docling_extractor import DoclingExtractor
from ingestion.processors.figure_processor import FigureProcessor

pdf = Path("data/page_cache/Q1FY26 PPT/page_17.pdf")

doc = DoclingExtractor().extract(pdf)

processor = FigureProcessor()

figures = processor.process(
    pdf,
    doc        # <-- not doc["document"]
)

print()

print("=" * 70)
print("FIGURES")
print("=" * 70)

for figure in figures:
    print(figure)