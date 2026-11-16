"""
Text Cleaner
------------

Production Version

Features:
✓ Cleans Unicode artifacts, non-breaking spaces, and Private Use Area (PUA) bullet icons
✓ Deduplicates doubled sentences caused by PDF text-layer overlaps (e.g. S + " " + S)
✓ Filters isolated slide numbers and stray 1-2 digit artifacts
✓ Normalizes whitespace and clean punctuation
"""

import re


class TextCleaner:

    @staticmethod
    def deduplicate_doubled_text(s: str) -> str:
        """Collapse duplicate text overlays where a sentence is repeated twice."""
        s = s.strip()
        if not s:
            return ""

        # Check word-halving for sentences with 6+ words
        words = s.split()
        n_words = len(words)
        if n_words >= 6 and n_words % 2 == 0:
            half = n_words // 2
            if words[:half] == words[half:]:
                return " ".join(words[:half])

        # Check character-halving (S + " " + S or S + S)
        n = len(s)
        for mid in range(n // 2 - 2, n // 2 + 2):
            if 0 < mid < n:
                left = s[:mid].strip()
                right = s[mid:].strip()
                if left and left.lower() == right.lower():
                    return left

        return s

    def clean_text_string(self, text: str) -> str:
        """Clean an individual text string."""
        if not text:
            return ""

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        # Replace Private Use Area (PUA) bullet/icon characters (common in PowerPoint exports)
        text = re.sub(r"[\ue000-\uf8ff]", "", text)

        # Clean common encoding artifacts like 
        text = text.replace("\ufffd", "-")

        # Deduplicate doubled sentences
        text = self.deduplicate_doubled_text(text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def clean(self, document):
        """Clean all pages and paragraphs in the document."""
        for page in document.pages:
            cleaned_paras = []
            seen_texts = set()

            for paragraph in page.paragraphs:
                text = self.clean_text_string(paragraph.text)
                if not text:
                    continue

                # Filter standalone 1-2 digit numbers (slide numbers / stray footnotes)
                if re.fullmatch(r"\d{1,2}", text):
                    continue

                low = text.lower()
                if low in seen_texts:
                    continue
                seen_texts.add(low)

                paragraph.text = text
                cleaned_paras.append(paragraph)

            page.paragraphs = cleaned_paras

        return document
