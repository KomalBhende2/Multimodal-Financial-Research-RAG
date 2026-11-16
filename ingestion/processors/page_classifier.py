"""
Page Layout Classifier
----------------------

Production Version

Classifies PDF pages into structural archetypes based on geometric properties:
1. SCANNED_RASTER: Low vector word density, high image area coverage.
2. STRUCTURED_2D_CANVAS: Slide cards, multi-column grids, financial snapshots, time-series charts.
3. FLOWING_NARRATIVE: Continuous paragraphs across standard single or dual columns.
"""

from enum import Enum
import re
import pymupdf


class PageType(Enum):
    SCANNED_RASTER = "scanned_raster"
    STRUCTURED_2D_CANVAS = "structured_2d_canvas"
    FLOWING_NARRATIVE = "flowing_narrative"


class PageLayoutClassifier:

    def __init__(self, gutter_threshold: float = 25.0):
        self.gutter_threshold = gutter_threshold

    def classify(self, pdf_page) -> PageType:
        """Classify a PyMuPDF page into an architectural layout type."""
        rect = pdf_page.rect
        page_w = rect.width
        page_h = rect.height
        page_area = page_w * page_h

        words = pdf_page.get_text("words")
        blocks = pdf_page.get_text("blocks")
        images = pdf_page.get_images(full=True)

        # --------------------------------------------------------
        # 1. Scanned / Raster Check
        # --------------------------------------------------------
        # Check total image coverage
        total_img_area = 0.0
        if images:
            for img_info in images:
                xref = img_info[0]
                img_rects = pdf_page.get_image_rects(xref)
                for r in img_rects:
                    total_img_area += r.width * r.height
        coverage = total_img_area / page_area if page_area > 0 else 0.0

        # a) If vector word count is very low (< 20) and image coverage is substantial
        if len(words) < 20 and images and coverage > 0.60:
            return PageType.SCANNED_RASTER

        # b) Synthetic / Corrupt Embedded OCR Layer Check:
        # Scanned documents often embed an invisible 'GlyphLessFont' or synthetic OCR text layer.
        # If a single image covers >= 85% of the page and uses synthetic OCR fonts,
        # it is a raster page with corrupt OCR, NOT native vector layout.
        if coverage >= 0.85 and images:
            fonts = pdf_page.get_fonts()
            font_names = [f[3].lower() for f in fonts] if fonts else []
            is_synthetic_ocr = any(
                "glyphless" in fn or "invisible" in fn or "ocr" in fn
                for fn in font_names
            )
            if is_synthetic_ocr:
                return PageType.SCANNED_RASTER

        # --------------------------------------------------------
        # 2. Structured 2D Canvas Check (Slides, Cards, Grids)
        # --------------------------------------------------------
        # A slide / 2D canvas is characterized by:
        # a) Landscape aspect ratio (width > height)
        # b) Presence of multi-column cards, 2D grids, or multi-period axes
        # c) Clear horizontal or vertical gutters
        is_landscape = page_w > page_h

        # Check for multi-period time series axis (e.g. Q1FY25 Q1FY26 Q1FY27)
        has_period_axis = any(
            len(re.findall(r"Q[1-4]FY\d{2}|FY\d{2}", b[4], re.I)) >= 2
            for b in blocks
        )
        if has_period_axis:
            return PageType.STRUCTURED_2D_CANVAS

        # Check for left-margin row labels + top column headers (2D Grid)
        left_labels = [b for b in blocks if b[0] < 120 and 240 <= b[1] <= 505 and re.search(r"[a-zA-Z]", b[4])]
        top_headers = [b for b in blocks if 220 <= b[1] <= 290 and b[0] >= 120 and re.search(r"[a-zA-Z]", b[4])]
        if len(left_labels) >= 3 and len(top_headers) >= 3:
            return PageType.STRUCTURED_2D_CANVAS

        # Check for multi-column cards via X-distribution
        body_words = [w for w in words if 120 <= (w[1] + w[3]) / 2 <= 500]
        if len(body_words) >= 15:
            # Project word centers along X-axis
            x_centers = sorted([(w[0] + w[2]) / 2 for w in body_words])
            # Find distinct clusters separated by gutters > 35pt
            clusters = []
            for xc in x_centers:
                if not clusters:
                    clusters.append([xc])
                else:
                    if xc - clusters[-1][-1] < 40.0:
                        clusters[-1].append(xc)
                    else:
                        clusters.append([xc])

            significant_clusters = [c for c in clusters if len(c) >= 5]
            if len(significant_clusters) >= 3:
                return PageType.STRUCTURED_2D_CANVAS

        # If landscape slide presentation with short bullet blocks
        if is_landscape and len(blocks) >= 4:
            avg_block_words = len(words) / max(len(blocks), 1)
            # Slides have short blocks (avg < 25 words per block)
            if avg_block_words < 25:
                return PageType.STRUCTURED_2D_CANVAS

        # --------------------------------------------------------
        # 3. Default: Flowing Narrative Document
        # --------------------------------------------------------
        return PageType.FLOWING_NARRATIVE
