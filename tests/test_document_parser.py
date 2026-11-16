from pathlib import Path

from ingestion.extractors.docling_extractor import DoclingExtractor
from ingestion.processors.document_parser import DocumentParser


page = Path("data/page_cache/Q1FY26 PPT/page_1.pdf")

extractor = DoclingExtractor()

doc = extractor.extract(page)

parser = DocumentParser()

parsed = parser.parse(
    doc["document"],
    filename=page.name
)

print("="*80)
print("Filename")
print("="*80)

print(parsed.filename)

print()

for page in parsed.pages:

    print("="*80)

    print("Page:", page.page_number)

    print("Title:", page.title)

    print("Paragraphs:", len(page.paragraphs))

    print("Tables:", len(page.tables))

    print("Figures:", len(page.figures))

    print()

    for p in page.paragraphs[:5]:

        print("-", p.text[:100])