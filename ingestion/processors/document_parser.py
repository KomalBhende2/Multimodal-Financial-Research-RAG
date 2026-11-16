"""
Production Document Parser
--------------------------

Converts DoclingDocument into our internal schema.

Preserves reading order and captures:
- Titles
- Headings
- Paragraphs
- Lists
- Tables
- Figures
"""

import sys

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from ingestion.models.document_schema import (
    Document,
    Page,
    Paragraph,
    Table,
    Figure,
)


class DocumentParser:

    @staticmethod
    def _is_valid_title(text: str) -> bool:
        if not text:
            return False
        t = text.strip()
        if len(t) < 3:
            return False
        letters = re.findall(r'[a-zA-Z]', t)
        if len(letters) < 3:
            return False
        if re.fullmatch(r'[\d\W_]+', t):
            return False
        return True

    def parse(self, doc, filename=""):

        document = Document(filename=filename)
        pages = {}
        total_pages = doc.num_pages()

        for page_no in range(1, total_pages + 1):
            pages[page_no] = Page(page_number=page_no)

        items_processed = 0

        # ----------------------------------------------------
        # Iterate through Docling items
        # ----------------------------------------------------
        for item, level in doc.iterate_items():
            items_processed += 1

            if not hasattr(item, "prov") or not item.prov:
                continue

            page_no = item.prov[0].page_no
            if page_no not in pages:
                continue

            page = pages[page_no]
            label = str(getattr(item, "label", "")).lower()
            text = getattr(item, "text", "") or ""
            text = text.strip()

            # Ignore empty text unless it's a table or picture
            if text == "" and "table" not in label and "picture" not in label:
                continue

            # ----------------------------------------------------
            # TITLE
            # ----------------------------------------------------
            if "title" in label:
                if not page.title:
                    page.title = text

                page.paragraphs.append(
                    Paragraph(
                        text=text,
                        heading_level=0,
                        source="docling"
                    )
                )
                continue

            # ----------------------------------------------------
            # HEADINGS
            # ----------------------------------------------------
            if (
                "section_header" in label
                or "heading" in label
                or "header" in label
            ):
                if not page.title and self._is_valid_title(text):
                    page.title = text

                page.paragraphs.append(
                    Paragraph(
                        text=text,
                        heading_level=1,
                        source="docling"
                    )
                )
                continue

            # ----------------------------------------------------
            # LIST ITEMS
            # ----------------------------------------------------
            if "list" in label:
                page.paragraphs.append(
                    Paragraph(
                        text=text,
                        heading_level=0,
                        source="docling"
                    )
                )
                continue

            # ----------------------------------------------------
            # NORMAL TEXT
            # ----------------------------------------------------
            if "text" in label or "paragraph" in label:
                if not page.title and len(text) < 120 and self._is_valid_title(text):
                    page.title = text

                page.paragraphs.append(
                    Paragraph(
                        text=text,
                        heading_level=0,
                        source="docling"
                    )
                )
                continue

            # ----------------------------------------------------
            # TABLE
            # ----------------------------------------------------
            if "table" in label:
                rows = []
                try:
                    # Pass doc to adhere to Docling 2.x API (eliminates deprecation warning)
                    try:
                        df = item.export_to_dataframe(doc=doc)
                    except TypeError:
                        df = item.export_to_dataframe()
                    rows = df.fillna("").values.tolist()
                except Exception:
                    pass

                page.tables.append(
                    Table(
                        title="",
                        rows=rows,
                        page_number=page_no,
                        source="docling"
                    )
                )
                continue

        document.pages = [pages[i] for i in sorted(pages)]
        return document
