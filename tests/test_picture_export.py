from pathlib import Path

import fitz

pdf = Path("data/page_cache/Q1FY26 PPT/page_1.pdf")

doc = fitz.open(pdf)

page = doc[0]

pix = page.get_pixmap(dpi=300)

pix.save("sample.png")

print("Saved sample.png")