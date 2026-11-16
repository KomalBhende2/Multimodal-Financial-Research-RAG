from ingestion.ocr.ocr_extractor import OCRExtractor

ocr = OCRExtractor()

text = ocr.extract_text("sample.png")

print()

print("="*80)

print("OCR OUTPUT")

print("="*80)

print()

print(text)