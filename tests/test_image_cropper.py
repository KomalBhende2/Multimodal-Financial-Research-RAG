from pathlib import Path

from ingestion.extractors.docling_extractor import DoclingExtractor
from ingestion.vision.image_cropper import ImageCropper

pdf = Path("data/page_cache/Q1FY26 PPT/page_17.pdf")

doc = DoclingExtractor().extract(pdf)

cropper = ImageCropper(dpi=300)

images = cropper.crop_images(
    pdf,
    doc
)

print()

for img in images:
    print(img)