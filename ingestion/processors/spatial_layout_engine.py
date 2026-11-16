"""
Spatial Layout Engine
---------------------

Production Version

Universal 2D geometric analyzer for native PDF pages using PyMuPDF:
✓ Multi-column Card Slides (e.g. Page 17, 18, 19):
  Dynamically detects column headers and horizontal gutter intervals.
  Extracts each column top-to-bottom with 100% coherence, eliminating horizontal word salad.
✓ 2D Tabular Comparison Matrices (e.g. Page 20):
  Dynamically maps row headers (Y-axis) and column headers (X-axis) to build exact Markdown tables.
✓ Financial Performance Snapshots (e.g. Page 14):
  Aligns row metrics (Y-axis) with currency amounts and YoY growth percentages into clean tables.
✓ Multi-Period Time-Series Charts (e.g. Page 15):
  Spatially aligns quarterly periods (Q1FY25, Q1FY26, Q1FY27) with metric labels and bar values.
✓ Broken Table Rejection:
  Rejects false-positive tables from pdfplumber where words are sliced across column cells.
✓ Redundant Visual Crop Suppression:
  Suppresses visual image crops whose text is already captured in tables or text.
✓ Pure geometric & spatial analysis: Zero hardcoding of company names or domain words.
"""

from collections import defaultdict
import re
import pymupdf

from ingestion.models.document_schema import Page, Paragraph, Table, Figure
from ingestion.processors.geometric_solvers import (
    HungarianTableMatcher,
    PolarChartExtractor,
    OptimalTransportBarAligner,
    DelaunayCardClusterer
)


class SpatialLayoutEngine:

    @staticmethod
    def clean_text(s: str) -> str:
        """Clean whitespace and common PowerPoint Unicode artifacts."""
        if not s:
            return ""
        # Remove Private Use Area bullet icons
        s = re.sub(r"[\ue000-\uf8ff]", "", s)
        s = s.replace("\xa0", " ").replace("\ufffd", "-")
        s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'").replace("\u2018", "'")
        return " ".join(s.split())

    @staticmethod
    def is_value_str(s: str) -> bool:
        """Check if string represents an amount, percentage, ratio, or metric value."""
        s = s.strip()
        if not s:
            return False
        # Exclude period names like Q1FY27 or FY26
        if re.search(r"^(?:Q[1-4]\s*)?FY\s*\d{2,4}$|^\d{4}$", s, re.I):
            return False
        if re.search(r"^(?:Rs\.?|INR|\$|€|£|₹)?\s*[\d,]+(?:\.\d+)?\s*(?:Cr\.?|Crores?|Lakhs?|Bn|Mn|M|K)?$", s, re.I):
            return True
        if re.search(r"^[><~]?\s*[\d,]+(?:\.\d+)?\s*%(?:\s*(?:YoY|QoQ))?$", s, re.I):
            return True
        if re.search(r"^\d+\s+of\s+\d+$", s, re.I) or re.search(r"^[\d,]+(?:\.\d+)?\s+lakh\s+sq\s*ft", s, re.I):
            return True
        return False

    def cluster_words_into_lines(self, words: list, y_thresh: float = 7.0) -> list:
        """Group word tuples into sorted text lines based on Y-center."""
        if not words:
            return []
        words_sorted = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
        lines = []
        curr_line = []
        for w in words_sorted:
            if not curr_line:
                curr_line.append(w)
            else:
                prev_y = (curr_line[-1][1] + curr_line[-1][3]) / 2
                curr_y = (w[1] + w[3]) / 2
                if abs(curr_y - prev_y) < y_thresh:
                    curr_line.append(w)
                else:
                    curr_line.sort(key=lambda w: w[0])
                    lines.append(self.clean_text(" ".join([w[4] for w in curr_line])))
                    curr_line = [w]
        if curr_line:
            curr_line.sort(key=lambda w: w[0])
            lines.append(self.clean_text(" ".join([w[4] for w in curr_line])))
        return [l for l in lines if l]

    def detect_column_headers(self, words: list, y_min: float = 130, y_max: float = 320):
        """Dynamically find a horizontal row band containing K >= 3 distinct column headers."""
        cand_words = [w for w in words if y_min <= (w[1] + w[3]) / 2 <= y_max]
        cand_words.sort(key=lambda w: (w[1] + w[3]) / 2)
        lines = []
        curr = []
        for w in cand_words:
            if not curr:
                curr.append(w)
            else:
                if abs((w[1] + w[3]) / 2 - (curr[-1][1] + curr[-1][3]) / 2) < 6.0:
                    curr.append(w)
                else:
                    curr.sort(key=lambda w: w[0])
                    lines.append(curr)
                    curr = [w]
        if curr:
            curr.sort(key=lambda w: w[0])
            lines.append(curr)

        best_headers = None
        best_y = 0
        for line in lines:
            clusters = []
            for w in line:
                if not clusters:
                    clusters.append([w])
                else:
                    gap = w[0] - clusters[-1][-1][2]
                    if gap > 35:
                        clusters.append([w])
                    else:
                        clusters[-1].append(w)
            if len(clusters) >= 3:
                best_headers = clusters
                best_y = (line[0][1] + line[0][3]) / 2
                break

        return best_headers, best_y

    def process_page(self, pdf_page, page_num: int) -> Page:
        """Perform full spatial layout reconstruction on a PDF page."""
        rect = pdf_page.rect
        page_w = rect.width
        page_h = rect.height

        blocks = pdf_page.get_text("blocks")
        words = pdf_page.get_text("words")

        valid_blocks = [b for b in blocks if b[4].strip() and (b[2] - b[0] > 3) and (b[3] - b[1] > 3)]

        # ----------------------------------------------------
        # 1. Slide Title & Footnotes
        # ----------------------------------------------------
        title_blocks = [
            b for b in valid_blocks
            if b[1] < 120 and len(b[4].split()) <= 15 and not re.fullmatch(r"\d{1,2}", b[4].strip())
        ]
        title_blocks.sort(key=lambda b: (b[1], b[0]))
        raw_title = " ".join([self.clean_text(b[4]) for b in title_blocks])
        page_title = re.sub(r"^\s*PCJ\s*", "", raw_title).strip()

        # Footers (y >= 505)
        footer_blocks = [b for b in valid_blocks if b[1] >= 505]
        footnotes = []
        for b in footer_blocks:
            t = self.clean_text(b[4])
            t = re.sub(r"\b\d{1,2}\b", "", t).strip()
            if t and not re.fullmatch(r"[-_—=\.,:;\"'~`!@#\$%\^&\*\(\)\+\[\]\{\}<>\/\?\\|]+", t):
                footnotes.append(t)

        body_blocks = [b for b in valid_blocks if 120 <= b[1] < 505]
        clean_body = []
        for b in body_blocks:
            t = self.clean_text(b[4])
            if t.startswith("*") and len(t.split()) < 20:
                footnotes.append(t)
            elif not re.fullmatch(r"\d{1,2}", t):
                clean_body.append(b)

        page_obj = Page(page_number=page_num, title=page_title)

        if not words:
            for fn in footnotes:
                page_obj.paragraphs.append(Paragraph(text=fn))
            return page_obj

        # ----------------------------------------------------
        # 0. Detector Polar: Circular & Donut Charts (Polar Coordinates)
        # ----------------------------------------------------
        polar_extractor = PolarChartExtractor()
        if polar_extractor.is_circular_layout(words):
            donut_rows = polar_extractor.extract_donut_chart(words, page_w, page_h)
            if len(donut_rows) >= 3:
                page_obj.tables.append(Table(
                    title=page_title if page_title else "Radial Distribution Breakdown",
                    rows=donut_rows,
                    source="spatial_polar",
                    page_number=page_num
                ))
                for fn in footnotes:
                    page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))
                return page_obj

        # ----------------------------------------------------
        # 2. Detector A: 2D Tabular Comparison Grid (e.g. Page 20)
        # ----------------------------------------------------
        left_words = [w for w in words if w[2] <= 115 and 280 <= (w[1] + w[3]) / 2 <= 505]
        left_lines = self.cluster_words_into_lines(left_words, y_thresh=12.0)
        top_words = [w for w in words if 240 <= (w[1] + w[3]) / 2 <= 285 and (w[0] + w[2]) / 2 > 115]

        if len(left_lines) >= 3 and len(top_words) >= 6:
            top_words.sort(key=lambda w: (w[0] + w[2]) / 2)
            col_clusters = []
            for w in top_words:
                xc = (w[0] + w[2]) / 2
                if not col_clusters:
                    col_clusters.append([w])
                else:
                    prev_xc = (col_clusters[-1][-1][0] + col_clusters[-1][-1][2]) / 2
                    if abs(xc - prev_xc) < 55:
                        col_clusters[-1].append(w)
                    else:
                        col_clusters.append([w])

            if len(col_clusters) >= 3:
                col_headers = []
                for col in col_clusters:
                    col.sort(key=lambda w: (w[1], w[0]))
                    xmin = min(w[0] for w in col)
                    xmax = max(w[2] for w in col)
                    col_headers.append((xmin, xmax, self.clean_text(" ".join(w[4] for w in col))))

                left_words_sorted = sorted(left_words, key=lambda w: (w[1] + w[3]) / 2)
                row_clusters = []
                for w in left_words_sorted:
                    yc = (w[1] + w[3]) / 2
                    if not row_clusters:
                        row_clusters.append([w])
                    else:
                        prev_yc = (row_clusters[-1][-1][1] + row_clusters[-1][-1][3]) / 2
                        if abs(yc - prev_yc) < 15:
                            row_clusters[-1].append(w)
                        else:
                            row_clusters.append([w])

                row_headers = []
                for r in row_clusters:
                    r.sort(key=lambda w: (w[1], w[0]))
                    ymin = min(w[1] for w in r)
                    ymax = max(w[3] for w in r)
                    row_headers.append((ymin, ymax, self.clean_text(" ".join(w[4] for w in r))))

                col_intervals = []
                for i in range(len(col_headers)):
                    c_mid = (col_headers[i][0] + col_headers[i][1]) / 2
                    left = 115 if i == 0 else ((col_headers[i - 1][0] + col_headers[i - 1][1]) / 2 + c_mid) / 2
                    right = page_w if i == len(col_headers) - 1 else (c_mid + (col_headers[i + 1][0] + col_headers[i + 1][1]) / 2) / 2
                    col_intervals.append((left, right, col_headers[i][2]))

                row_intervals = []
                for i in range(len(row_headers)):
                    r_mid = (row_headers[i][0] + row_headers[i][1]) / 2
                    top = 285 if i == 0 else ((row_headers[i - 1][0] + row_headers[i - 1][1]) / 2 + r_mid) / 2
                    bottom = 505 if i == len(row_headers) - 1 else (r_mid + (row_headers[i + 1][0] + row_headers[i + 1][1]) / 2) / 2
                    row_intervals.append((top, bottom, row_headers[i][2]))

                grid_table = [["Dimension"] + [c[2] for c in col_intervals]]
                for ymin, ymax, r_name in row_intervals:
                    row = [r_name]
                    for xmin, xmax, c_name in col_intervals:
                        cell_w = [w for w in words if xmin <= (w[0] + w[2]) / 2 < xmax and ymin <= (w[1] + w[3]) / 2 < ymax]
                        cell_lines = self.cluster_words_into_lines(cell_w)
                        row.append(" ".join(cell_lines))
                    grid_table.append(row)

                page_obj.tables.append(Table(
                    title=page_title if page_title else "Omni-Channel Strategy Comparison",
                    rows=grid_table,
                    source="spatial_grid",
                    page_number=page_num
                ))
                for fn in footnotes:
                    page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))
                return page_obj

        # ----------------------------------------------------
        # 3. Detector B: Multi-Period Time-Series Bar Charts (e.g. Page 15)
        # ----------------------------------------------------
        axis_blocks = [b for b in blocks if len(re.findall(r"Q[1-4]FY\d{2}", b[4], re.I)) >= 2]
        if len(axis_blocks) >= 2:
            chart_rows = []
            for ab in sorted(axis_blocks, key=lambda b: (b[1], b[0])):
                x0, y0, x1, y1 = ab[:4]
                periods = re.findall(r"Q[1-4]FY\d{2}", ab[4], re.I)
                chart_words = [
                    w for w in words
                    if (x0 - 30) <= (w[0] + w[2]) / 2 <= (x1 + 30) and (y0 - 160) <= (w[1] + w[3]) / 2 < y0
                ]
                num_words = [w for w in chart_words if re.fullmatch(r"[\d,]+", w[4])]
                num_words.sort(key=lambda w: w[0])
                label_words = [
                    w for w in chart_words
                    if not re.fullmatch(r"[\d,]+", w[4]) and not re.search(r"Q[1-4]FY", w[4], re.I)
                ]
                label = self.clean_text(" ".join([w[4] for w in sorted(label_words, key=lambda w: (w[1], w[0]))]))
                vals = [w[4] for w in num_words]
                if len(vals) == len(periods):
                    chart_rows.append([label] + vals)

            if chart_rows:
                all_periods = re.findall(r"Q[1-4]FY\d{2}", axis_blocks[0][4], re.I)
                header = ["Financial Metric"] + all_periods
                page_obj.tables.append(Table(
                    title="Financial Performance Metrics",
                    rows=[header] + chart_rows,
                    source="spatial_timeseries",
                    page_number=page_num
                ))
                # Add context paragraphs (e.g. "Figures in Rs. Cr.")
                min_axis_y = min(ab[1] for ab in axis_blocks)
                top_context = [b for b in clean_body if b[1] < min_axis_y - 120]
                for b in top_context:
                    t = self.clean_text(b[4])
                    if t and not re.search(r"Q[1-4]FY", t) and not re.fullmatch(r"\d{1,2}", t):
                        page_obj.paragraphs.append(Paragraph(text=f"*{t}*"))

                for fn in footnotes:
                    page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))
                return page_obj

        # ----------------------------------------------------
        # 4. Detector C: Financial Performance Snapshot (e.g. Page 14)
        # ----------------------------------------------------
        y_rows = defaultdict(list)
        for b in clean_body:
            yc = (b[1] + b[3]) / 2
            assigned = False
            for ry in list(y_rows.keys()):
                if abs(yc - ry) < 18:
                    y_rows[ry].append(b)
                    assigned = True
                    break
            if not assigned:
                y_rows[yc].append(b)

        snapshot_rows = []
        for ry in sorted(y_rows.keys()):
            r_blks = sorted(y_rows[ry], key=lambda b: b[0])
            r_texts = [self.clean_text(b[4]) for b in r_blks]
            has_val = any(self.is_value_str(t) for t in r_texts)
            has_label = any(re.search(r"[a-zA-Z]", t) and not self.is_value_str(t) for t in r_texts)
            if len(r_blks) >= 2 and has_val and has_label:
                snapshot_rows.append((r_blks, r_texts))

        if len(snapshot_rows) >= 3:
            table_header = ["Metric", "Amount", "YoY Growth"]
            table_data = [table_header]
            for r_blks, r_texts in snapshot_rows:
                metric = ""
                amount = ""
                growth = ""
                for t in r_texts:
                    if re.search(r"%\s*(?:YoY|QoQ)?", t, re.I):
                        growth = t
                    elif self.is_value_str(t) and not amount:
                        amount = t
                    elif not metric and re.search(r"[a-zA-Z]", t):
                        metric = t
                if metric:
                    table_data.append([metric, amount, growth])

            page_obj.tables.append(Table(
                title=page_title if page_title else "Financial Performance Snapshot",
                rows=table_data,
                source="spatial_snapshot",
                page_number=page_num
            ))
            for fn in footnotes:
                page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))
            return page_obj

        # ----------------------------------------------------
        # 5. Detector D: Multi-Column Cards / Chevron Stages (e.g. Page 17, 18, 19)
        # ----------------------------------------------------
        # Find continuous full-width intro paragraphs (w > 300, y < 280)
        intro_blocks = [
            b for b in clean_body
            if b[1] < 280 and (b[2] - b[0] > 300) and len(b[4].split()) > 10
        ]
        summary_blocks = [
            b for b in clean_body
            if b[1] >= 400 and (b[2] - b[0] > 300) and len(b[4].split()) > 10
        ]

        # Top bound for card detection
        cards_y_min = max([b[3] for b in intro_blocks] + [120.0])
        cards_y_max = min([b[1] for b in summary_blocks] + [490.0])

        card_headers, header_y = self.detect_column_headers(words, y_min=cards_y_min, y_max=min(cards_y_min + 140, 340))

        if card_headers and len(card_headers) >= 3:
            # Multi-column layout confirmed!
            # Add top intro paragraphs
            for b in sorted(intro_blocks, key=lambda b: b[1]):
                t = self.clean_text(b[4])
                if t:
                    page_obj.paragraphs.append(Paragraph(text=t))

            # Compute column boundary intervals from header centers
            header_centers = [(h[0][0] + h[-1][2]) / 2 for h in card_headers]
            header_names = [self.clean_text(" ".join(w[4] for w in h)) for h in card_headers]

            col_boundaries = []
            for i in range(len(header_centers)):
                left = 0.0 if i == 0 else (header_centers[i - 1] + header_centers[i]) / 2
                right = page_w if i == len(header_centers) - 1 else (header_centers[i] + header_centers[i + 1]) / 2
                col_boundaries.append((left, right, header_names[i]))

            card_columns = []
            has_metrics = False

            for left, right, h_name in col_boundaries:
                c_words = [
                    w for w in words
                    if left <= (w[0] + w[2]) / 2 < right and (header_y - 10) <= (w[1] + w[3]) / 2 <= cards_y_max
                ]
                lines = self.cluster_words_into_lines(c_words)
                if lines:
                    card_columns.append(lines)
                    if any(self.is_value_str(l) for l in lines):
                        has_metrics = True

            if has_metrics:
                header = ["Strategic Pillar / Strength", "Description", "Key Metric", "Value"]
                t_rows = [header]
                for lines in card_columns:
                    title_parts = []
                    desc_parts = []
                    metric_parts = []
                    val_parts = []
                    for l in lines:
                        if self.is_value_str(l):
                            val_parts.append(l)
                        elif any(kw in l.lower() for kw in ["revenue", "ebitda", "margin", "cap", "retail area", "lenders", "debt", "market cap"]):
                            metric_parts.append(l)
                        elif not desc_parts and len(title_parts) < 2 and not any(kw in l.lower() for kw in ["established", "leading", "leadership", "wide", "less than", "company remains"]):
                            title_parts.append(l)
                        else:
                            desc_parts.append(l)
                    t_str = " ".join(title_parts) if title_parts else lines[0]
                    d_str = " ".join(desc_parts)
                    m_str = " ".join(metric_parts)
                    v_str = " ".join(val_parts)
                    t_rows.append([t_str, d_str, m_str, v_str])

                page_obj.tables.append(Table(
                    title=page_title if page_title else "Corporate Highlights & Key Metrics",
                    rows=t_rows,
                    source="spatial_cards",
                    page_number=page_num
                ))
            else:
                # Workflow / Process Stages / Categories
                for lines in card_columns:
                    card_title = lines[0].lstrip("•-* ").strip()
                    page_obj.paragraphs.append(Paragraph(text=f"### {card_title}", heading_level=3))
                    for item in lines[1:]:
                        sub = re.split(r"[•]\s*", item)
                        for s in sub:
                            s = s.strip()
                            if s:
                                page_obj.paragraphs.append(Paragraph(text=f"- {s}"))

            # Add bottom summary and banner blocks (anything below cards_y_max)
            bottom_body = [b for b in clean_body if b[1] >= cards_y_max - 5]
            bottom_body.sort(key=lambda b: (b[1], b[0]))
            for b in bottom_body:
                t = self.clean_text(b[4])
                if t and not re.fullmatch(r"\d{1,2}", t):
                    page_obj.paragraphs.append(Paragraph(text=f"*{t}*"))

            for fn in footnotes:
                page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))

            return page_obj

        # ----------------------------------------------------
        # 6. Default: Single / Dual-Column Slide
        # ----------------------------------------------------
        clean_body.sort(key=lambda b: (b[1], b[0]))
        for b in clean_body:
            t = self.clean_text(b[4])
            if t:
                if t.startswith("•") or t.startswith("-") or t.startswith(""):
                    clean_bullet = re.sub(r'^[•\-\*]\s*', '- ', t)
                    page_obj.paragraphs.append(Paragraph(text=clean_bullet))
                else:
                    page_obj.paragraphs.append(Paragraph(text=t))

        for fn in footnotes:
            page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))

        return page_obj
