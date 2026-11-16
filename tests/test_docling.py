from pathlib import Path
from docling.document_converter import DocumentConverter

pdf = Path("data/documents/Q1FY26 PPT.pdf")

converter = DocumentConverter()

doc = converter.convert(pdf).document

print("="*80)
print("TOTAL TABLES")
print("="*80)

print(len(doc.tables))

if len(doc.tables):

    table = doc.tables[0]

    print("\nTYPE")
    print(type(table))

    print("\nATTRIBUTES")

    for attr in dir(table):

        if attr.startswith("_"):
            continue

        try:

            value = getattr(table, attr)

            if callable(value):
                continue

            print(attr, ":", value)

        except:
            pass