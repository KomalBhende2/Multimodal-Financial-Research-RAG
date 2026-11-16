"""
OCR Extractor
-------------

Production Version (Silent & Generic)

- Memory optimized
- RapidOCR 3.x
- Suppresses false alarms ("text detection result is empty" on pure images/jewellery/diagrams)
- Automatic text cleaning
"""

from pathlib import Path
import logging
import gc

from rapidocr import RapidOCR


class OCRExtractor:

    def __init__(self):

        print("Loading OCR Engine...")

        # Initialize RapidOCR engine
        self.engine = RapidOCR()

        # Suppress RapidOCR internal false-alarm warnings (e.g. empty detection on photos/logos)
        self.logger = logging.getLogger("RapidOCR")
        self.logger.setLevel(logging.ERROR)

        print("OCR Ready.\n")

    # ---------------------------------------------------------

    def extract_text(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            return ""

        # Ensure warning spam is muted during OCR extraction of graphical crops
        self.logger.setLevel(logging.ERROR)

        try:
            result = self.engine(str(image_path))
        except Exception:
            return ""

        if result is None:
            gc.collect()
            return ""

        text = ""

        # -----------------------------------------------------
        # RapidOCR 3.x (RapidOCROutput object)
        # -----------------------------------------------------
        if hasattr(result, "txts") and result.txts:
            text = "\n".join(result.txts)

        # -----------------------------------------------------
        # RapidOCR tuple result [(box, text, score), ...]
        # -----------------------------------------------------
        elif isinstance(result, (list, tuple)):
            res_items = result[0] if (result and isinstance(result[0], (list, tuple))) else result
            lines = []
            if res_items:
                for item in res_items:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lines.append(str(item[1]))
                    elif isinstance(item, str):
                        lines.append(item)
            text = "\n".join(lines)

        # -----------------------------------------------------
        # Clean text
        # -----------------------------------------------------
        cleaned = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Skip standalone single character noise unless alphanumeric
            if len(line) == 1 and not line.isalnum():
                continue

            cleaned.append(line)

        gc.collect()

        return "\n".join(cleaned)
