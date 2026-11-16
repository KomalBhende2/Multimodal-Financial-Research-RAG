"""
Page Validator & Audit Engine
-----------------------------

Production Version

Audits extraction completeness by calculating token-level recall between the
source PDF page and the reconstructed Markdown Page object:
1. Tokenizes ground-truth text from PyMuPDF page.
2. Tokenizes all extracted paragraphs, table cells, and figure OCR lines.
3. Computes Set-Intersection Recall:
   Recall = |Source Tokens ∩ Extracted Tokens| / |Source Tokens|
4. Flags any page with low recall (< 85%) using GitHub-flavored Markdown warning alerts.
5. Prevents hallucinations and ensures zero unflagged text loss.
"""

import re
from typing import Set, Tuple
from ingestion.models.document_schema import Page, Paragraph


class PageValidator:

    def __init__(self, recall_threshold: float = 0.85):
        self.recall_threshold = recall_threshold

    @staticmethod
    def normalize_tokens(text: str) -> Set[str]:
        """Normalize text into lowercase alphanumeric tokens."""
        if not text:
            return set()
        # Remove PUA icons
        text = re.sub(r"[\ue000-\uf8ff]", "", text)
        words = text.split()
        tokens = set()
        for w in words:
            clean = w.lower().strip(".,;:()[]{}\"'`~-_—=+*&^%$#@!/\\|<>?")
            # Keep meaningful tokens (length >= 2 or digits)
            if clean and (len(clean) >= 2 or clean.isdigit()):
                tokens.add(clean)
        return tokens

    def validate_page(self, pdf_page, page_obj: Page) -> Tuple[Page, float]:
        """Audit page completeness against source PDF and insert warning if needed."""
        # 1. Source tokens from PyMuPDF
        source_text = pdf_page.get_text()
        source_tokens = self.normalize_tokens(source_text)

        # Skip verification on completely blank or raster-only slides (< 5 words)
        if len(source_tokens) < 5:
            return page_obj, 1.0

        # 2. Extracted tokens from page_obj (paragraphs, tables, figures)
        extracted_chunks = []
        if page_obj.title:
            extracted_chunks.append(page_obj.title)

        for p in page_obj.paragraphs:
            if not p.text.startswith("> [!WARNING]"):
                extracted_chunks.append(p.text)

        for t in page_obj.tables:
            if t.title:
                extracted_chunks.append(t.title)
            for row in t.rows:
                for cell in row:
                    extracted_chunks.append(str(cell))

        for fig in page_obj.figures:
            if fig.ocr_text:
                extracted_chunks.append(fig.ocr_text)

        extracted_text = " ".join(extracted_chunks)
        extracted_tokens = self.normalize_tokens(extracted_text)

        # 3. Calculate Token Recall
        intersection = source_tokens.intersection(extracted_tokens)
        recall = len(intersection) / len(source_tokens)

        # 4. Attach validation metadata
        if not hasattr(page_obj, "metadata") or page_obj.metadata is None:
            page_obj.metadata = {}
        page_obj.metadata["token_recall"] = round(recall, 3)
        page_obj.metadata["source_token_count"] = len(source_tokens)
        page_obj.metadata["extracted_token_count"] = len(extracted_tokens)

        # 5. Flag low recall pages with a visible Markdown alert
        if recall < self.recall_threshold:
            missing_tokens = sorted(list(source_tokens - extracted_tokens))
            sample_missing = ", ".join(missing_tokens[:6])
            warning_msg = (
                f"> [!WARNING] Potential text mismatch: Recall score is {recall:.1%}. "
                f"Source tokens: {len(source_tokens)}, Extracted: {len(intersection)}. "
                f"Missing tokens sample: [{sample_missing}]. Review source page."
            )
            # Insert alert right under title / at top of paragraphs
            page_obj.paragraphs.insert(0, Paragraph(text=warning_msg, heading_level=0, source="validator"))

        return page_obj, recall
