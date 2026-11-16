"""
Structured Data Reconstructor
-----------------------------

Production Version

Features:
✓ Reconstructs fragmented/unstructured OCR figures into clean Markdown tables:
  - Parallel figure columns (e.g. Metric Names + Amounts + YoY Growth)
✓ Reconstructs sequential KPI cards from reading-order text streams:
  - Grouped values followed by grouped labels (e.g. At a Glance cards)
  - Alternating Metric-Value-Growth card sequences (with figure cross-correlation to fill missing amounts)
  - Stat / Target cards (e.g. Debt reduction milestones)
  - Timeline & event sequences
✓ Suppresses consumed visual crops and deduplicates matching text from paragraphs
✓ Formats footnotes neatly under reconstructed tables
✓ Zero hardcoding of company names or domain keywords
"""

from collections import defaultdict
import re
from rapidfuzz import fuzz

from ingestion.models.document_schema import Table


class StructuredDataReconstructor:

    @staticmethod
    def is_value_str(s: str) -> bool:
        """Check if string represents a financial value, currency amount, percentage, or statistic."""
        s = s.strip()
        if not s:
            return False
        # Period strings like Q1FY27 or FY26 are periods, not amounts
        if re.search(r"^(?:Q[1-4]\s*)?FY\s*\d{2,4}$", s, re.I):
            return False
        # Currency / amounts (e.g. Rs. 877 Cr., $450M, €100, 3,353 Crores, 877)
        if re.search(r"^(?:Rs\.?|INR|\$|€|£|₹)?\s*[\d,]+(?:\.\d+)?\s*(?:Cr\.?|Crores?|Lakhs?|Bn|Mn|M|K)?$", s, re.I):
            return True
        # Percentages / changes (e.g. 21% YoY, 81% YoY, 20%, >96%, <4%)
        if re.search(r"^[><~]?\s*[\d,]+(?:\.\d+)?\s*%(?:\s*(?:YoY|QoQ))?$", s, re.I):
            return True
        # Ratios / fractions / measurements (e.g. 7 of 14, 2.05 lakh sq ft)
        if re.search(r"^\d+\s+of\s+\d+$", s, re.I) or re.search(r"^[\d,]+(?:\.\d+)?\s+lakh\s+sq\s*ft", s, re.I):
            return True
        return False

    @staticmethod
    def is_percentage_str(s: str) -> bool:
        """Check if string is specifically a percentage or growth metric."""
        return bool(re.search(r"^[><~]?\s*[\d,]+(?:\.\d+)?\s*%", s.strip(), re.I))

    @staticmethod
    def is_metric_label(s: str) -> bool:
        """Check if string is a concise metric label/parameter name."""
        s = s.strip()
        if not s or len(s) > 55:
            return False
        # Skip period labels (e.g. Q1FY25, FY26)
        if re.search(r"^(?:Q[1-4]\s*)?FY\s*\d{2,4}$|^\d{4}$", s, re.I):
            return False
        if StructuredDataReconstructor.is_value_str(s):
            return False
        if not re.search(r"[a-zA-Z]", s):
            return False
        if s.startswith("*"):
            return False
        if s.endswith(".") and len(s.split()) > 6:
            return False
        return True

    @staticmethod
    def clean_ocr_lines(text: str) -> list:
        """Extract clean lines from figure OCR text, stripping noise and slide numbers."""
        lines = []
        seen = set()
        for l in text.splitlines():
            l = l.strip()
            if not l:
                continue
            # Skip isolated 1-2 digit slide numbers or footnote indexes
            if re.fullmatch(r"\d{1,2}", l):
                continue
            # Skip pure punctuation noise
            if re.fullmatch(r"[-_—=\.,:;\"'~`!@#\$%\^&\*\(\)\+\[\]\{\}<>\/\?\\|]+", l):
                continue
            low = l.lower()
            if low in seen:
                continue
            seen.add(low)
            lines.append(l)
        return lines

    def reconstruct(self, page):
        """Analyze page figures and paragraphs to reconstruct structured tables and clean noise."""
        # If page already has pristine spatial tables/layout, preserve it
        if any(t.source.startswith("spatial_") for t in page.tables):
            return page

        consumed_texts = set()
        new_tables = []
        consumed_fig_indices = set()

        # ----------------------------------------------------
        # 1. Parallel Figures Reconstruction (e.g. Page 14)
        # ----------------------------------------------------
        if page.figures and len(page.figures) >= 2:
            fig_data = []
            for idx, fig in enumerate(page.figures):
                lines = self.clean_ocr_lines(fig.ocr_text)
                if lines:
                    fig_data.append((idx, fig, lines))

            if len(fig_data) >= 2:
                by_len = defaultdict(list)
                for idx, fig, lines in fig_data:
                    by_len[len(lines)].append((idx, fig, lines))

                for line_count, group in by_len.items():
                    if 2 <= line_count <= 12 and 2 <= len(group) <= 4:
                        has_labels = any(all(self.is_metric_label(x) for x in lines) for _, _, lines in group)
                        has_values = any(any(self.is_value_str(x) for x in lines) for _, _, lines in group)

                        if has_labels and has_values:
                            def col_priority(g_item):
                                lines = g_item[2]
                                if all(self.is_metric_label(x) for x in lines):
                                    return 1
                                if all(self.is_value_str(x) and not self.is_percentage_str(x) for x in lines):
                                    return 2
                                if any(self.is_percentage_str(x) for x in lines):
                                    return 3
                                return 4

                            sorted_group = sorted(group, key=col_priority)

                            headers = []
                            for idx, (_, _, lines) in enumerate(sorted_group):
                                if all(self.is_metric_label(x) for x in lines):
                                    headers.append("Metric")
                                elif all(self.is_value_str(x) and not self.is_percentage_str(x) for x in lines):
                                    headers.append("Amount")
                                elif any(self.is_percentage_str(x) for x in lines):
                                    headers.append("YoY Growth")
                                else:
                                    headers.append(f"Col {idx+1}")

                            table_rows = [headers]
                            for row_idx in range(line_count):
                                row = [g[2][row_idx] for g in sorted_group]
                                table_rows.append(row)
                                for cell in row:
                                    consumed_texts.add(cell.lower())
                                    consumed_texts.add(cell.strip("*").strip().lower())

                            for g_idx, _, _ in sorted_group:
                                consumed_fig_indices.add(g_idx)

                            p_title = page.title.strip() if page.title else ""
                            table_title = p_title if p_title else "Financial Performance Snapshot"

                            new_tables.append(
                                Table(title=table_title, rows=table_rows, source="parallel_figures", page_number=page.page_number)
                            )

        # ----------------------------------------------------
        # 2. Sequential Metric Cards in Paragraphs (e.g. Page 17, Page 10)
        # ----------------------------------------------------
        paras = [p.text.strip() for p in page.paragraphs if p.text.strip()]

        # Pattern A: Grouped values followed by grouped labels (e.g. Page 17)
        for i in range(len(paras)):
            val_run = []
            j = i
            while j < len(paras) and self.is_value_str(paras[j]):
                val_run.append(paras[j])
                j += 1
            K = len(val_run)
            if 3 <= K <= 7:
                label_run = []
                while j < len(paras) and self.is_metric_label(paras[j]) and len(label_run) < K:
                    label_run.append(paras[j])
                    j += 1
                if len(label_run) == K:
                    table_rows = [["Metric", "Value"]]
                    for l, v in zip(label_run, val_run):
                        table_rows.append([l, v])
                        consumed_texts.add(l.lower())
                        consumed_texts.add(v.lower())
                    new_tables.append(
                        Table(title="Key Metrics", rows=table_rows, source="grouped_kpis", page_number=page.page_number)
                    )
                    break

        # Pattern B: Alternating Metric-Value-Growth (e.g. Page 14)
        if not new_tables:
            i = 0
            card_rows = []
            while i < len(paras):
                p0 = paras[i]
                if self.is_metric_label(p0):
                    val = ""
                    growth = ""
                    step = 1
                    if i + 1 < len(paras):
                        p1 = paras[i+1]
                        if self.is_value_str(p1) and not self.is_percentage_str(p1):
                            val = p1
                            step = 2
                            if i + 2 < len(paras) and self.is_percentage_str(paras[i+2]):
                                growth = paras[i+2]
                                step = 3
                        elif self.is_percentage_str(p1):
                            growth = p1
                            step = 2
                    if val or growth:
                        card_rows.append([p0, val, growth])
                        i += step
                    else:
                        i += 1
                else:
                    i += 1

            if len(card_rows) >= 2:
                # Check if figures on the page contain matching amounts to fill in missing values
                for f_idx, fig in enumerate(page.figures):
                    fig_lines = self.clean_ocr_lines(fig.ocr_text)
                    if len(fig_lines) == len(card_rows) and all(self.is_value_str(x) and not self.is_percentage_str(x) for x in fig_lines):
                        # Match! Fill missing amounts
                        for r_idx, r in enumerate(card_rows):
                            if not r[1] or r[1] == "-":
                                r[1] = fig_lines[r_idx]
                        consumed_fig_indices.add(f_idx)
                        break

                headers = ["Metric", "Amount", "Change / Growth"] if any(r[2] for r in card_rows) else ["Metric", "Amount"]
                table_rows = [headers]
                for r in card_rows:
                    if len(headers) == 3:
                        table_rows.append(r)
                    else:
                        table_rows.append([r[0], r[1]])
                    for c in r:
                        if c:
                            consumed_texts.add(c.lower())
                            consumed_texts.add(c.strip("*").strip().lower())

                p_title = page.title.strip() if page.title else "Financial Highlights"
                new_tables.append(
                    Table(title=p_title, rows=table_rows, source="alternating_kpis", page_number=page.page_number)
                )

        # Pattern C: Stat / Target Cards (e.g. Page 10)
        stat_card_pattern = re.compile(r"^[><~]?\s*\d+(?:\.\d+)?%?$|^\d+\s+of\s+\d+$|^[><~]?\d+%$", re.I)
        stat_cards = []
        for i in range(len(paras)):
            s = paras[i].strip()
            if stat_card_pattern.match(s):
                if i + 1 < len(paras) and self.is_metric_label(paras[i+1]):
                    title = paras[i+1].strip()
                    desc = paras[i+2].strip() if (i + 2 < len(paras) and len(paras[i+2].split()) >= 4 and not stat_card_pattern.match(paras[i+2])) else ""
                    stat_cards.append((title, s, desc))
        if len(stat_cards) >= 2:
            table_rows = [["Metric / Target", "Value", "Details"]]
            for t, val, d in stat_cards:
                table_rows.append([t, val, d])
                consumed_texts.add(t.lower())
                consumed_texts.add(val.lower())
                if d:
                    consumed_texts.add(d.lower())
            new_tables.append(
                Table(title="Key Milestones & Targets", rows=table_rows, source="stat_cards", page_number=page.page_number)
            )

        # Pattern D: Chronological Milestones (e.g. Page 10 Timeline)
        date_pattern = re.compile(r"^(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})$", re.I)
        milestones = []
        for i in range(len(paras)):
            s = paras[i].strip()
            if date_pattern.match(s):
                if i + 1 < len(paras) and len(paras[i+1].split()) >= 3 and not date_pattern.match(paras[i+1]):
                    milestones.append((s, paras[i+1].strip()))
        if len(milestones) >= 3:
            table_rows = [["Milestone / Date", "Event & Resolution"]]
            for d, event in milestones:
                table_rows.append([d, event])
                consumed_texts.add(d.lower())
                consumed_texts.add(event.lower())
            new_tables.append(
                Table(title="Resolution Timeline", rows=table_rows, source="timeline", page_number=page.page_number)
            )

        # ----------------------------------------------------
        # 3. Apply Reconstructed Tables & Clean Page
        # ----------------------------------------------------
        if new_tables:
            page.tables.extend(new_tables)

        if consumed_fig_indices:
            page.figures = [f for idx, f in enumerate(page.figures) if idx not in consumed_fig_indices]

        # Clean paragraphs
        cleaned_paras = []
        p_title_low = page.title.strip().lower() if page.title else ""
        for p in page.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            txt_low = txt.lower()
            txt_clean = txt.strip("*").strip().lower()

            if re.fullmatch(r"\d{1,2}", txt):
                continue
            if p_title_low and (txt_low == p_title_low or fuzz.ratio(txt_low, p_title_low) > 90):
                continue
            if txt_low in consumed_texts or txt_clean in consumed_texts:
                continue
            cleaned_paras.append(p)

        page.paragraphs = cleaned_paras
        return page
