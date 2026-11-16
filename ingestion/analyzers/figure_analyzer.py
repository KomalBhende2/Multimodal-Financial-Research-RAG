"""
Figure Analyzer
---------------

Production Version (Generic)

Classifies figures into meaningful categories for
Institutional Research RAG.

Categories
----------
financial_chart
table_image
diagram
product_image
ui_screenshot
logo
icon
text_block
unknown
"""

from pathlib import Path
from PIL import Image
import gc

from ingestion.ocr.ocr_extractor import OCRExtractor


class FigureAnalyzer:

    def __init__(self):

        print("Figure Analyzer Ready.\n")

        self.ocr = OCRExtractor()

    # ----------------------------------------------------------

    def analyze(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            return None

        with Image.open(image_path) as image:

            width, height = image.size

        ocr_text = self.ocr.extract_text(image_path)

        ocr_text = "\n".join(

            line.strip()

            for line in ocr_text.splitlines()

            if line.strip()

        )

        figure_type = self.classify(

            width,
            height,
            ocr_text

        )

        gc.collect()

        return {

            "image_path": str(image_path),

            "width": width,

            "height": height,

            "ocr_text": ocr_text,

            "figure_type": figure_type

        }

    # ----------------------------------------------------------

    def classify(

        self,
        width,
        height,
        text

    ):

        text = text.lower().strip()

        # --------------------------------------------
        # Empty
        # --------------------------------------------

        if text == "":
            return "icon"

        # --------------------------------------------
        # Large text screenshot
        # --------------------------------------------

        if len(text) > 700:
            return "text_block"

        # --------------------------------------------
        # Financial Charts
        # --------------------------------------------

        financial_keywords = [

            "revenue",
            "income",
            "sales",
            "profit",
            "ebitda",
            "ebit",
            "pat",
            "gross margin",
            "margin",
            "cash",
            "cash flow",
            "assets",
            "liabilities",
            "equity",
            "inventory",
            "capex",
            "opex",
            "working capital",
            "market share",
            "volume",
            "units sold",
            "deliveries",
            "growth",
            "yoy",
            "qoq",
            "fy",
            "q1",
            "q2",
            "q3",
            "q4",
            "%",
            "mn",
            "crore",
            "bps"

        ]

        if any(word in text for word in financial_keywords):
            return "financial_chart"

        # --------------------------------------------
        # Tables
        # --------------------------------------------

        table_keywords = [

            "note:",
            "source:",
            "amount",
            "total",
            "average",
            "year",
            "quarter",
            "particulars"

        ]

        if any(word in text for word in table_keywords):

            return "table_image"

        # --------------------------------------------
        # Diagrams / Workflows / Architecture
        # --------------------------------------------

        diagram_keywords = [

            "ecosystem",
            "platform",
            "workflow",
            "architecture",
            "process",
            "network",
            "lifecycle",
            "patent",
            "manufacturing",
            "technology",
            "supply chain",
            "framework"

        ]

        if any(word in text for word in diagram_keywords):

            return "diagram"

        # --------------------------------------------
        # Mobile / Web UI Screenshots
        # --------------------------------------------

        ui_keywords = [

            "screen",
            "app",
            "mobile",
            "interface",
            "dashboard",
            "login",
            "settings",
            "portal",
            "website",
            "button",
            "click"

        ]

        if any(word in text for word in ui_keywords):

            return "ui_screenshot"

        # --------------------------------------------
        # Product Images & Catalogs
        # --------------------------------------------

        product_keywords = [

            "collection",
            "design",
            "product",
            "specifications",
            "model",
            "material",
            "edition",
            "series",
            "dimensions"

        ]

        if any(word in text for word in product_keywords):

            return "product_image"

        # --------------------------------------------
        # Logos
        # --------------------------------------------

        if len(text) < 8:

            return "logo"

        # --------------------------------------------
        # Tiny crops
        # --------------------------------------------

        if width < 120 or height < 120:

            return "icon"

        return "unknown"
