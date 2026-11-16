"""
Deduplicator
------------

Production Version

Features:
✓ Deduplicates tables across multiple extractors (Docling vs pdfplumber)
✓ Filters out phantom/single-cell/diagonal-empty tables from slide boxes
✓ Sanitizes excessively long table titles
✓ Suppresses figures whose OCR text is already present in page paragraphs or tables
✓ Removes paragraphs duplicated inside tables or matching page titles
✓ Removes duplicate paragraphs using fuzzy string matching
✓ Removes duplicate figures on the same page
"""

from rapidfuzz import fuzz
import re


class Deduplicator:

    def _table_to_text(self, table) -> str:
        """Convert table rows to a normalized string for comparison."""
        lines = []
        for row in table.rows:
            clean_cells = [str(c).strip().lower() for c in row if str(c).strip()]
            if clean_cells:
                lines.append(" ".join(clean_cells))
        return "\n".join(lines).strip()

    def _clean_table_rows(self, table):
        """Clean rows, remove duplicate rows and empty cells."""
        seen_rows = set()
        cleaned_rows = []
        for row in table.rows:
            cleaned_cell_list = [
                re.sub(r'\s+', ' ', str(cell)).strip() if cell is not None else ""
                for cell in row
            ]
            # Skip rows with no content
            if not any(cleaned_cell_list):
                continue
            row_key = tuple(cleaned_cell_list)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            cleaned_rows.append(cleaned_cell_list)

        table.rows = cleaned_rows
        return table

    def _is_valid_table(self, table) -> bool:
        """Filter out phantom/dummy tables created from visual boxes on slides."""
        if not table.rows or len(table.rows) < 2:
            return False

        # If title is excessively long (> 70 chars and > 10 words), sanitize or reject
        title = (table.title or "").strip()
        if len(title) > 70 and len(title.split()) > 10:
            # Check if it looks like paragraph text dumped into table title
            if len(table.rows) <= 2:
                return False
            # Otherwise sanitize title
            table.title = title.split(".")[0][:50].strip()

        # Count non-empty cells
        total_non_empty = sum(len([c for c in r if str(c).strip()]) for r in table.rows)
        if total_non_empty < 3:
            return False

        # Check for diagonal-empty 2-row tables (e.g. [['', 'Header'], ['Bullet items', '']])
        if len(table.rows) == 2:
            r0_cells = len([c for c in table.rows[0] if str(c).strip()])
            r1_cells = len([c for c in table.rows[1] if str(c).strip()])
            if r0_cells <= 1 and r1_cells <= 1:
                return False

        return True

    def _deduplicate_tables(self, existing_tables, recovered_tables):
        """Merge and deduplicate tables from Docling and pdfplumber."""
        all_tables = []

        # Clean and filter existing tables
        for t in existing_tables:
            t = self._clean_table_rows(t)
            if self._is_valid_table(t):
                all_tables.append(t)

        # Merge recovered tables without adding duplicates
        for t in recovered_tables:
            t = self._clean_table_rows(t)
            if not self._is_valid_table(t):
                continue

            t_text = self._table_to_text(t)
            if not t_text or len(t_text) < 20:
                continue

            duplicate = False
            for idx, existing in enumerate(all_tables):
                existing_text = self._table_to_text(existing)
                sim = fuzz.ratio(t_text, existing_text)
                if sim > 75 or (len(t_text) > 40 and t_text in existing_text) or (len(existing_text) > 40 and existing_text in t_text):
                    duplicate = True
                    existing_cells = sum(len([c for c in r if c]) for r in existing.rows)
                    recovered_cells = sum(len([c for c in r if c]) for r in t.rows)
                    if recovered_cells > existing_cells:
                        all_tables[idx] = t
                    break

            if not duplicate:
                all_tables.append(t)

        return all_tables

    def merge(self, document, recovered_tables):

        for page_idx, page in enumerate(document.pages, start=1):

            # --------------------------------
            # 1. Deduplicate Tables
            # --------------------------------
            extra_tables = recovered_tables.get(page_idx, [])
            page.tables = self._deduplicate_tables(page.tables, extra_tables)

            # Build a set of table cell texts for this page to avoid duplicating them as paragraphs
            table_cell_texts = set()
            for t in page.tables:
                for row in t.rows:
                    for cell in row:
                        c_str = cell.strip().lower()
                        if len(c_str) > 2:
                            table_cell_texts.add(c_str)

            # --------------------------------
            # 2. Deduplicate Paragraphs
            # --------------------------------
            unique_paragraphs = []
            page_title_lower = page.title.strip().lower() if page.title else ""

            for paragraph in page.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue

                text_lower = text.lower()

                # Skip if text is identical or nearly identical to page title
                if page_title_lower and (text_lower == page_title_lower or fuzz.ratio(text_lower, page_title_lower) > 90):
                    continue

                # Skip if text is identical to a standalone cell in a table on this page
                if text_lower in table_cell_texts and len(text.split()) < 6:
                    continue

                # Fuzzy duplicate check against already accepted paragraphs
                duplicate = False
                for old_text in unique_paragraphs:
                    if text_lower == old_text.lower() or fuzz.ratio(text, old_text) > 90:
                        duplicate = True
                        break

                if not duplicate:
                    unique_paragraphs.append(text)
                    paragraph.text = text
                else:
                    paragraph.text = ""  # Mark for removal

            page.paragraphs = [p for p in page.paragraphs if p.text]

            # --------------------------------
            # 3. Deduplicate Figures
            # --------------------------------
            unique_figures = []
            seen_fig_texts = []
            all_page_text = " ".join([p.text.lower() for p in page.paragraphs] + list(table_cell_texts))

            for fig in page.figures:
                fig_text = fig.ocr_text.strip().lower()
                if not fig_text:
                    continue

                # Check if figure text is already predominantly present in page paragraphs / tables
                fig_words = [w.strip(".,;:()[]{}") for w in fig_text.split() if len(w.strip(".,;:()[]{}")) >= 2]
                if fig_words:
                    words_in_page = sum(1 for w in fig_words if w in all_page_text)
                    # If 50%+ of words are already written in paragraphs or tables, suppress redundant figure
                    if words_in_page / len(fig_words) >= 0.5:
                        continue

                # Check similarity against other figures on this page
                is_dup = False
                for existing_txt in seen_fig_texts:
                    if fuzz.ratio(fig_text, existing_txt) > 85:
                        is_dup = True
                        break

                if not is_dup:
                    seen_fig_texts.append(fig_text)
                    unique_figures.append(fig)

            page.figures = unique_figures

        return document
