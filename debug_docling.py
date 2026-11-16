from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = "data/documents/Q1FY26 PPT.pdf"   # <-- change if needed

print("Loading Docling...")
try:
    # Docling 2.x: OCR is enabled automatically if available
    converter = DocumentConverter()
    print("[OK] Docling loaded (OCR will be used if available)\n")
except Exception as e:
    print(f"[ERROR] Failed to load Docling: {e}\n")
    exit(1)

result = converter.convert(pdf_path)
doc = result.document

print(f"Document has {doc.num_pages()} pages\n")
print("=" * 60)
print("ANALYZING PAGES 6-11")
print("=" * 60 + "\n")

for page in range(6, 11):

    print("\n" + "="*60)
    print("PAGE", page)
    print("="*60)

    found = False
    items_with_prov = 0
    items_without_prov = 0

    for item, level in doc.iterate_items():

        # Count all items regardless of provenance
        if hasattr(item, "prov") and item.prov:
            items_with_prov += 1
            if item.prov[0].page_no != page:
                continue
        else:
            items_without_prov += 1
            continue

        found = True

        print(f"\n[FOUND] Item found on page {page}")
        print(f"  Type    : {type(item).__name__}")
        print(f"  Label   : {getattr(item, 'label', 'N/A')}")
        text = getattr(item, "text", "")
        text_preview = text[:100] if text else "[EMPTY]"
        print(f"  Text    : {text_preview}")

    print(f"\nPage {page} Summary:")
    print(f"  Items with provenance: {items_with_prov}")
    print(f"  Items without provenance (skipped): {items_without_prov}")
    
    if not found:
        print("  [EMPTY] NO ITEMS FOUND ON THIS PAGE")
        print("  --> This page may be scanned/image-based")
        print("  --> Text extraction failed - OCR may not have worked")