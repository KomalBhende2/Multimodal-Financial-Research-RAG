"""
Image Cropper
-------------

Memory Optimized + Figure Merging

Instead of saving every PictureItem separately,
nearby picture regions are merged into one crop.

This greatly improves:

✓ OCR quality
✓ Figure context
✓ Less duplicate figures
✓ Better RAG retrieval
✓ Fast CPU processing (150 DPI)
"""

from pathlib import Path
import pymupdf as fitz


class ImageCropper:

    def __init__(self, dpi=150, merge_distance=35):

        self.dpi = dpi
        self.merge_distance = merge_distance

        print(f"Image Cropper Ready ({dpi} DPI)\n")

    # ----------------------------------------------------
    # Merge nearby rectangles
    # ----------------------------------------------------

    def merge_rectangles(self, rects):

        changed = True

        while changed:

            changed = False
            merged = []

            while rects:

                current = rects.pop(0)

                i = 0

                while i < len(rects):

                    other = rects[i]

                    expanded = fitz.Rect(
                        current.x0 - self.merge_distance,
                        current.y0 - self.merge_distance,
                        current.x1 + self.merge_distance,
                        current.y1 + self.merge_distance,
                    )

                    if expanded.intersects(other):

                        current |= other

                        rects.pop(i)

                        changed = True

                    else:

                        i += 1

                merged.append(current)

            rects = merged

        return rects

    # ----------------------------------------------------
    # Extract Figures
    # ----------------------------------------------------

    def extract(self, pdf_path, docling_document):

        pdf_path = Path(pdf_path)

        output_folder = (
            Path("data")
            / "extracted_figures"
            / pdf_path.stem
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        pdf = fitz.open(pdf_path)

        zoom = self.dpi / 72

        matrix = fitz.Matrix(zoom, zoom)

        page_rects = {}

        # -----------------------------------------
        # Collect rectangles page-wise
        # -----------------------------------------

        for picture in docling_document.pictures:

            if not picture.prov:
                continue

            prov = picture.prov[0]

            page_no = prov.page_no

            page = pdf.load_page(page_no - 1)

            page_height = page.rect.height

            bbox = prov.bbox

            rect = fitz.Rect(
                bbox.l,
                page_height - bbox.t,
                bbox.r,
                page_height - bbox.b,
            )

            if rect.is_empty:
                continue

            if rect.width < 8 or rect.height < 8:
                continue

            page_rects.setdefault(page_no, []).append(rect)

        # -----------------------------------------
        # Merge rectangles
        # -----------------------------------------

        saved = []

        for page_no, rects in page_rects.items():

            merged = self.merge_rectangles(rects)

            page = pdf.load_page(page_no - 1)

            for index, rect in enumerate(merged, start=1):

                pix = page.get_pixmap(
                    matrix=matrix,
                    clip=rect,
                    alpha=False
                )

                image_path = (
                    output_folder
                    / f"page_{page_no}_figure_{index}.png"
                )

                pix.save(image_path)

                saved.append(
                    {
                        "page": page_no,
                        "path": str(image_path),
                        "bbox": rect,
                    }
                )

        pdf.close()

        print(f"Saved {len(saved)} merged figures")

        return saved
