"""
Document Builder
----------------

Production Version (Clean, Structured & Complete Markdown Generator)

Features:
✓ Integrates StructuredDataReconstructor to transform parallel figures & KPI cards into clean tables
✓ Formats valid, professional GitHub-flavored Markdown tables with aligned headers
✓ Automatically groups and formats bullet lists, subheaders, and narrative paragraphs
✓ Neatly formats footnotes under their corresponding tables
✓ Sanitizes figure headings (never uses raw numbers/percentages as headings)
✓ Zero truncation of OCR data
✓ Zero hardcoded company or domain keywords
"""

from pathlib import Path
import re

from ingestion.processors.structured_reconstructor import StructuredDataReconstructor


class DocumentBuilder:

    def __init__(self):
        self.reconstructor = StructuredDataReconstructor()

    def clean_ocr(self, text: str) -> str:
        """Clean noisy OCR text generically without domain-specific hardcoding."""
        if not text:
            return ""

        lines = []
        seen = set()

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Skip standalone 1-2 digit slide numbers or footnote indexes
            if re.fullmatch(r"\d{1,2}", line):
                continue

            lower = line.lower()

            # Deduplicate repeated lines
            if lower in seen:
                continue
            seen.add(lower)

            # Skip standalone single character noise or pure punctuation
            if len(line) <= 1 and not line.isalnum():
                continue

            # Skip lines consisting purely of punctuation / non-alphanumeric noise
            if re.fullmatch(r"[-_—=\.,:;\"'~`!@#\$%\^&\*\(\)\+\[\]\{\}<>\/\?\\|]+", line):
                continue

            lines.append(line)

        return "\n".join(lines)

    def _clean_table_columns(self, rows: list) -> list:
        """Remove columns that are completely blank across all rows."""
        if not rows:
            return []

        max_cols = max(len(r) for r in rows)
        padded = [row + [""] * (max_cols - len(row)) for row in rows]

        # Determine non-empty columns
        non_empty_indices = [
            col_idx for col_idx in range(max_cols)
            if any(str(padded[r_idx][col_idx]).strip() for r_idx in range(len(padded)))
        ]

        if not non_empty_indices:
            return []

        cleaned_rows = [
            [padded[r_idx][c] for c in non_empty_indices]
            for r_idx in range(len(padded))
        ]
        return cleaned_rows

    def _format_markdown_table(self, rows: list) -> str:
        """Render rows as a valid Markdown table with proper header separator."""
        cleaned_rows = self._clean_table_columns(rows)
        if not cleaned_rows or len(cleaned_rows) < 2:
            return ""

        num_cols = len(cleaned_rows[0])

        # If it's a 1-column table that's just a bulleted list, format as a Markdown list
        if num_cols == 1:
            list_lines = []
            for row in cleaned_rows:
                val = str(row[0]).strip()
                if not val:
                    continue
                if val.startswith("•") or val.startswith("-"):
                    list_lines.append(val)
                else:
                    list_lines.append(f"- {val}")
            return "\n".join(list_lines) + "\n"

        # Format as Markdown table
        normalized_rows = []
        for row in cleaned_rows:
            clean_row = [
                str(cell).strip().replace("\n", " ").replace("|", "\\|")
                for cell in row
            ]
            normalized_rows.append(clean_row)

        header = normalized_rows[0]
        # Ensure header isn't completely blank
        if not any(header):
            header = [f"Col {i+1}" for i in range(num_cols)]

        output_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join([":---"] * num_cols) + " |"
        ]

        for row in normalized_rows[1:]:
            output_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(output_lines) + "\n"

    @staticmethod
    def _is_valid_page_title(title: str) -> bool:
        if not title:
            return False
        t = title.strip()
        if len(t) < 3:
            return False
        letters = re.findall(r'[a-zA-Z]', t)
        if len(letters) < 3:
            return False
        if re.fullmatch(r'[\d\W_]+', t):
            return False
        return True

    def build(self, document, output_folder="ingestion/output") -> Path:
        """Build and save structured Markdown (.md) document."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        stem = Path(document.filename).stem
        output_file = output_folder / f"{stem}.md"

        company_name = getattr(document, "company_name", getattr(document, "company", "Company"))
        doc_type = getattr(document, "document_type", "Document")
        fiscal_year = getattr(document, "fiscal_year", "")
        quarter = getattr(document, "quarter", "")

        with open(output_file, "w", encoding="utf-8") as f:

            # ---------------- Document Header ----------------
            f.write(f"# Document Extraction: {company_name}\n\n")

            f.write("## Document Overview\n")
            f.write(f"- **Company**: {company_name}\n")
            f.write(f"- **Document Type**: {doc_type}\n")
            if fiscal_year or quarter:
                period = f"{quarter} {fiscal_year}".strip()
                f.write(f"- **Reporting Period**: {period}\n")
            f.write(f"- **Total Pages**: {len(document.pages)}\n")
            f.write(f"- **Source File**: {Path(document.filename).name}\n\n")
            f.write("---\n\n")

            # ---------------- Page Processing ----------------
            for page in document.pages:

                # Run intelligent reconstruction for tables, KPI cards, and parallel figures
                page = self.reconstructor.reconstruct(page)

                page_title = page.title.strip() if page.title else ""
                if page_title and self._is_valid_page_title(page_title):
                    f.write(f"# Page {page.page_number}: {page_title}\n\n")
                else:
                    # Check if first body paragraph is a clean short heading
                    candidate_title = ""
                    for p in page.paragraphs:
                        p_t = p.text.strip()
                        if self._is_valid_page_title(p_t) and len(p_t) < 70 and not p_t.startswith("*") and not p_t.startswith(">"):
                            candidate_title = p_t
                            break
                    if candidate_title:
                        f.write(f"# Page {page.page_number}: {candidate_title}\n\n")
                    else:
                        f.write(f"# Page {page.page_number}\n\n")

                # ---------------- Paragraphs / Text ----------------
                seen_paras = set()
                body_paras = []
                footnotes = []

                for para in page.paragraphs:
                    text = para.text.strip()
                    if not text:
                        continue

                    lower = text.lower()
                    if lower in seen_paras:
                        continue
                    seen_paras.add(lower)

                    # Skip title duplicate
                    if page_title and lower == page_title.lower():
                        continue

                    # Separate footnotes starting with *
                    if text.startswith("*"):
                        footnotes.append(text)
                    else:
                        body_paras.append(text)

                # Format body paragraphs and bullet lists
                if body_paras:
                    i = 0
                    while i < len(body_paras):
                        text = body_paras[i]

                        # If text already has markdown heading level (e.g. ### Card Title), output directly
                        if text.startswith("#"):
                            f.write(f"{text}\n\n")
                            i += 1
                            continue

                        # If text is a blockquote or alert banner (e.g. > [!WARNING]), output directly
                        if text.startswith(">"):
                            f.write(f"{text}\n\n")
                            i += 1
                            continue

                        # If text is already a formatted bullet point, output cleanly
                        if text.startswith("•") or text.startswith("- ") or text.startswith(""):
                            clean_bullet = re.sub(r'^[•\-\*]\s*', '- ', text)
                            f.write(f"{clean_bullet}\n")
                            i += 1
                            continue

                        # Check if text is a category heading followed by short items
                        list_items = []
                        j = i + 1
                        while j < len(body_paras):
                            next_t = body_paras[j]
                            if next_t.startswith("•") or next_t.startswith("- ") or next_t.startswith(""):
                                list_items.append(re.sub(r'^[•\-\*]\s*', '', next_t))
                                j += 1
                            elif len(next_t.split()) <= 6 and not next_t.endswith(".") and not next_t.startswith("#"):
                                list_items.append(next_t)
                                j += 1
                            else:
                                break

                        if len(list_items) >= 2 and len(text.split()) <= 5 and not text.endswith("."):
                            clean_h = text.strip("*")
                            f.write(f"### {clean_h}\n\n")
                            for it in list_items:
                                f.write(f"- {it}\n")
                            f.write("\n")
                            i = j
                        else:
                            if text.startswith("•") or text.startswith("- ") or text.startswith(""):
                                clean_bullet = re.sub(r'^[•\-\*]\s*', '- ', text)
                                f.write(f"{clean_bullet}\n")
                            elif len(text.split()) <= 4 and not text.endswith("."):
                                f.write(f"**{text}**\n\n")
                            else:
                                f.write(f"{text}\n\n")
                            i += 1
                    f.write("\n")

                # ---------------- Tables ----------------
                if page.tables:
                    for t_idx, table in enumerate(page.tables, start=1):
                        title = table.title.strip() if table.title else ""
                        if title and title != "Recovered Table" and len(title) <= 60:
                            f.write(f"### Table: {title}\n\n")
                        table_md = self._format_markdown_table(table.rows)
                        if table_md.strip():
                            f.write(table_md)
                            f.write("\n")

                # ---------------- Footnotes ----------------
                if footnotes:
                    for fn in footnotes:
                        clean_fn = fn.strip("* ").strip()
                        if clean_fn:
                            f.write(f"*{clean_fn}*\n\n")

                # ---------------- Figures & Visuals ----------------
                if page.figures:
                    for f_idx, fig in enumerate(page.figures, start=1):
                        clean_ocr = self.clean_ocr(fig.ocr_text)
                        if not clean_ocr:
                            continue

                        fig_lines = clean_ocr.splitlines()
                        if not fig_lines:
                            continue

                        first_line = fig_lines[0].strip()

                        # If first line is a number, percentage, or currency, don't use it as a header
                        if self.reconstructor.is_value_str(first_line):
                            fig_header = "Visual Content / Metrics"
                            lines_to_write = fig_lines
                        else:
                            fig_header = first_line
                            lines_to_write = fig_lines[1:]

                        if fig.figure_type == "financial_chart":
                            f.write(f"### Chart Data: {fig_header}\n\n")
                        elif fig.figure_type == "diagram":
                            f.write(f"### Diagram / Workflow: {fig_header}\n\n")
                        else:
                            f.write(f"### Visual Content: {fig_header}\n\n")

                        # Write lines
                        for line in lines_to_write:
                            if line.startswith("•") or line.startswith("-") or line.startswith(""):
                                f.write(f"- {line.lstrip('•-* ').strip()}\n")
                            else:
                                f.write(f"- {line}\n")
                        f.write("\n")

                f.write("---\n\n")

        print(f"\n[SAVED] Output Markdown : {output_file}")
        return output_file
