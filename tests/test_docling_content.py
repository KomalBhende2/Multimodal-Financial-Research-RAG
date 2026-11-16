from pathlib import Path

from docling.document_converter import DocumentConverter

pdf = Path("data/documents/Q1FY26 PPT.pdf")

converter = DocumentConverter()

result = converter.convert(pdf)

doc = result.document

print("=" * 80)
print("Pages")
print("=" * 80)

print(doc.num_pages)

print("\n")

print("=" * 80)
print("Texts")
print("=" * 80)

print(type(doc.texts))
print("Count:", len(doc.texts))

print("\n")

print("=" * 80)
print("Tables")
print("=" * 80)

print(type(doc.tables))
print("Count:", len(doc.tables))

print("\n")

print("=" * 80)
print("Pictures")
print("=" * 80)

print(type(doc.pictures))
print("Count:", len(doc.pictures))

print("\n")

print("=" * 80)
print("First Text")
print("=" * 80)

if len(doc.texts):

    first = doc.texts[0]

    print(type(first))

    print(dir(first))