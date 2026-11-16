from pathlib import Path
from pprint import pprint

from ingestion.extractors.docling_extractor import DoclingExtractor

page = Path("data/page_cache/Q1FY26 PPT/page_1.pdf")

extractor = DoclingExtractor()

doc = extractor.extract(page)

print(type(doc))

print()

print("Available attributes:")

pprint(dir(doc))