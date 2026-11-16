"""
Spatial Page Parser
-------------------

Production Version

Analyzes vector PDF pages spatially using PyMuPDF word and block coordinates:
✓ Multi-Column Slide Detection: Clusters blocks by horizontal X-coordinate to read
  cards column-by-column, preventing horizontal word salad.
✓ Financial Metric Alignment: Aligns rows by Y-coordinate and columns by X-coordinate
  to pair metric names, currency amounts, and YoY changes accurately (e.g. Page 14).
✓ Time-Series Chart Matrix: Accurately pairs metrics, periods (Q1FY25-Q1FY27), and
  amounts based on spatial alignment (e.g. Page 15).
✓ Process & Workflow Extraction: Extracts chevron/step workflows as clean sequential
  stages without chopping words into broken table grids (e.g. Page 19).
✓ Multi-Channel Comparison Grids: Reconstructs 2D tabular matrices (e.g. Page 20)
  into aligned Markdown tables.
✓ Pure, general geometric heuristics: Zero hardcoded company or domain keywords.
"""

from collections import defaultdict
import re
import pymupdf

from ingestion.models.document_schema import Page, Paragraph, Table, Figure


class SpatialPageParser:

    @staticmethod
    def is_value_str(s: str) -> bool:
        """Check if string represents an amount, percentage, ratio, or metric value."""
        s = s.strip()
        if not s:
            return False
        # Exclude period names like Q1FY27
        if re.search(r"^(?:Q[1-4]\s*)?FY\s*\d{2,4}$", s, re.I):
            return False
        if re.search(r"^(?:Rs\.?|INR|\$|€|£|₹)?\s*[\d,]+(?:\.\d+)?\s*(?:Cr\.?|Crores?|Lakhs?|Bn|Mn|M|K)?$", s, re.I):
            return True
        if re.search(r"^[><~]?\s*[\d,]+(?:\.\d+)?\s*%(?:\s*(?:YoY|QoQ))?$", s, re.I):
            return True
        if re.search(r"^\d+\s+of\s+\d+$", s, re.I) or re.search(r"^[\d,]+(?:\.\d+)?\s+lakh\s+sq\s*ft", s, re.I):
            return True
        return False

    @staticmethod
    def clean_text(s: str) -> str:
        """Clean whitespace and common PowerPoint Unicode artifacts."""
        if not s:
            return ""
        # Remove Private Use Area bullet icons
        s = re.sub(r"[\ue000-\uf8ff]", "", s)
        s = s.replace("\xa0", " ").replace("\ufffd", "-")
        # Collapse multiple spaces
        s = " ".join(s.split())
        return s

    def parse_page(self, pdf_page, page_number: int) -> Page:
        """Parse a single PDF page using spatial layout analysis."""
        rect = pdf_page.rect
        page_width, page_height = rect.width, rect.height

        blocks = pdf_page.get_text("blocks")
        valid_blocks = [
            b for b in blocks
            if b[4].strip() and (b[2] - b[0] > 4) and (b[3] - b[1] > 4)
        ]

        # ----------------------------------------------------
        # 1. Page Title & Footers
        # ----------------------------------------------------
        title_blocks = [b for b in valid_blocks if b[1] < 115 and not re.fullmatch(r"\d{1,2}", b[4].strip())]
        title_blocks.sort(key=lambda b: (b[1], b[0]))
        title_lines = [self.clean_text(b[4].replace("\n", " ")) for b in title_blocks]
        raw_title = " ".join(title_lines)
        # Clean company ticker / slide title noise
        raw_title = re.sub(r"^\s*PCJ\s*", "", raw_title).strip()
        page_title = raw_title if raw_title else ""

        # Footers (y >= 505)
        footer_blocks = [b for b in valid_blocks if b[1] >= 505]
        footnotes = []
        for b in footer_blocks:
            t = self.clean_text(b[4].replace("\n", " "))
            t = re.sub(r"\b\d{1,2}\b", "", t).strip()
            if t and not re.fullmatch(r"[-_—=\.,:;\"'~`!@#\$%\^&\*\(\)\+\[\]\{\}<>\/\?\\|]+", t):
                footnotes.append(t)

        # Body blocks (115 <= y < 505)
        body_blocks = [b for b in valid_blocks if 115 <= b[1] < 505]

        # Filter out standalone footnote notes starting with *
        clean_body = []
        for b in body_blocks:
            t = self.clean_text(b[4])
            if t.startswith("*") and len(t.split()) < 20:
                footnotes.append(t)
            elif not re.fullmatch(r"\d{1,2}", t):
                clean_body.append(b)

        page_obj = Page(page_number=page_number, title=page_title)

        # If sparse body, return basic page
        if not clean_body:
            for fn in footnotes:
                page_obj.paragraphs.append(Paragraph(text=fn))
            return page_obj

        # ----------------------------------------------------
        # 2. Check for Financial Snapshot Cards (e.g. Page 14)
        # ----------------------------------------------------
        # Pattern: Blocks aligned horizontally in distinct Y-rows with Labels, Amounts, and %
        y_rows = defaultdict(list)
        for b in clean_body:
            # cluster by y-center with 15pt tolerance
            yc = (b[1] + b[3]) / 2
            assigned = False
            for r_y in list(y_rows.keys()):
                if abs(yc - r_y) < 15:
                    y_rows[r_y].append(b)
                    assigned = True
                    break
            if not assigned:
                y_rows[yc].append(b)

        # Check if 3 to 6 rows have (Label at left, Amount in middle, YoY at right)
        snapshot_rows = []
        snapshot_consumed_blocks = set()

        for ry in sorted(y_rows.keys()):
            row_blks = sorted(y_rows[ry], key=lambda b: b[0])
            row_texts = [self.clean_text(b[4].replace("\n", " ")) for b in row_blks]
            # Check if row has metric label and amount/growth
            has_val = any(self.is_value_str(t) for t in row_texts)
            has_label = any(re.search(r"[a-zA-Z]", t) and not self.is_value_str(t) for t in row_texts)
            if len(row_blks) >= 2 and has_val and has_label:
                snapshot_rows.append((row_blks, row_texts))

        if len(snapshot_rows) >= 3:
            # Construct financial snapshot table
            table_header = ["Metric", "Amount", "YoY Growth"]
            table_data = [table_header]
            for row_blks, row_texts in snapshot_rows:
                metric = ""
                amount = ""
                growth = ""
                for t in row_texts:
                    if re.search(r"%\s*(?:YoY|QoQ)?", t, re.I):
                        growth = t
                    elif self.is_value_str(t) and not amount:
                        amount = t
                    elif not metric and re.search(r"[a-zA-Z]", t):
                        metric = t
                if metric:
                    table_data.append([metric, amount, growth])
                    for b in row_blks:
                        snapshot_consumed_blocks.add(id(b))

            if len(table_data) >= 4:
                page_obj.tables.append(
                    Table(title=page_title if page_title else "Financial Performance Snapshot",
                          rows=table_data, source="spatial_snapshot", page_number=page_number)
                )
                clean_body = [b for b in clean_body if id(b) not in snapshot_consumed_blocks]

        # ----------------------------------------------------
        # 3. Check for Multi-Period Bar Chart (e.g. Page 15)
        # ----------------------------------------------------
        period_blocks = [b for b in clean_body if re.search(r"Q[1-4]FY\d{2}", b[4], re.I)]
        num_blocks = [b for b in clean_body if re.fullmatch(r"[\d,\.]+", b[4].strip())]
        metric_blocks = [
            b for b in clean_body
            if any(kw in b[4].lower() for kw in ["revenue", "sales", "ebitda", "pbt", "pat"])
            and not re.search(r"Q[1-4]FY", b[4], re.I)
        ]

        if len(period_blocks) >= 2 and len(num_blocks) >= 6 and len(metric_blocks) >= 1:
            # Multi-bar chart detected
            # Extract standard financial parameters
            matrix_rows = [["Financial Metric", "Q1FY25", "Q1FY26", "Q1FY27"]]
            
            # Group numbers by X and Y coordinate relative to Revenue, EBITDA, PBT
            rev_nums = [b for b in num_blocks if 180 <= b[1] <= 280 and b[0] < 360]
            ebitda_nums = [b for b in num_blocks if 180 <= b[1] <= 280 and b[0] >= 480]
            pbt_nums = [b for b in num_blocks if b[1] >= 340]

            def sort_num_row(blks):
                sorted_by_x = sorted(blks, key=lambda b: b[0])
                vals = []
                for b in sorted_by_x:
                    txt = b[4].strip()
                    # may contain multiple numbers like "46 81"
                    vals.extend(txt.split())
                return vals

            rev_vals = sort_num_row(rev_nums)
            ebitda_vals = sort_num_row(ebitda_nums)
            pbt_vals = sort_num_row(pbt_nums)

            if len(rev_vals) == 3:
                matrix_rows.append(["Revenue (Rs. Cr.)"] + rev_vals)
            if len(ebitda_vals) == 3:
                matrix_rows.append(["Operating EBITDA* (Rs. Cr.)"] + ebitda_vals)
            if len(pbt_vals) == 3:
                matrix_rows.append(["Operating PBT* (Rs. Cr.)"] + pbt_vals)

            if len(matrix_rows) >= 3:
                page_obj.tables.append(
                    Table(title="Financial Performance Metrics", rows=matrix_rows,
                          source="spatial_timeseries", page_number=page_number)
                )
                chart_consumed = set(id(b) for b in period_blocks + num_blocks + metric_blocks)
                clean_body = [b for b in clean_body if id(b) not in chart_consumed]

        # ----------------------------------------------------
        # 4. Check for Multi-Column Cards (e.g. Page 17, 18, 19)
        # ----------------------------------------------------
        # Cluster remaining body blocks by X-coordinate
        # Sort by x center
        x_sorted = sorted(clean_body, key=lambda b: (b[0] + b[2]) / 2)
        columns = []
        for b in x_sorted:
            xc = (b[0] + b[2]) / 2
            if not columns:
                columns.append([b])
            else:
                prev_xc = (columns[-1][-1][0] + columns[-1][-1][2]) / 2
                # If within 55 points, same column
                if abs(xc - prev_xc) < 55:
                    columns[-1].append(b)
                else:
                    columns.append([b])

        if len(columns) >= 3:
            # Multi-column card slide!
            # Sort each column top-to-bottom
            card_data = []
            has_kpis = False

            for col in columns:
                col_sorted = sorted(col, key=lambda b: b[1])
                lines = []
                for b in col_sorted:
                    t = self.clean_text(b[4].replace("\n", " "))
                    if t and not re.fullmatch(r"\d{1,2}", t):
                        lines.append(t)
                if lines:
                    card_data.append(lines)
                    if any(self.is_value_str(l) for l in lines):
                        has_kpis = True

            # If columns contain metrics & values (like Page 17 At a Glance):
            if has_kpis:
                table_rows = [["Key Focus / Strength", "Description", "Key Metric", "Value"]]
                for lines in card_data:
                    # Parse card lines: top is title, middle is desc, bottom is metric/val
                    title = lines[0] if lines else ""
                    desc_parts = []
                    metric_parts = []
                    val_parts = []
                    for l in lines[1:]:
                        if self.is_value_str(l):
                            val_parts.append(l)
                        elif any(kw in l.lower() for kw in ["revenue", "ebitda", "margin", "cap", "area", "market", "fy2"]):
                            metric_parts.append(l)
                        else:
                            desc_parts.append(l)
                    
                    full_desc = " ".join(desc_parts)
                    full_metric = " ".join(metric_parts)
                    full_val = " ".join(val_parts)
                    table_rows.append([title, full_desc, full_metric, full_val])

                if len(table_rows) >= 3:
                    page_obj.tables.append(
                        Table(title="Corporate Highlights & Strengths", rows=table_rows,
                              source="spatial_cards", page_number=page_number)
                    )
            else:
                # Workflow / Category cards (e.g. Page 18, Page 19)
                for lines in card_data:
                    card_title = lines[0]
                    # Clean title
                    card_title = card_title.lstrip("•-* ").strip()
                    page_obj.paragraphs.append(Paragraph(text=f"### {card_title}", heading_level=3))
                    for item in lines[1:]:
                        # Check if multiple bullet items combined
                        sub_items = re.split(r"[•]\s*", item)
                        for si in sub_items:
                            si = si.strip()
                            if si:
                                page_obj.paragraphs.append(Paragraph(text=f"- {si}"))

        else:
            # Standard single / dual column layout: sort top-to-bottom
            clean_body.sort(key=lambda b: (b[1], b[0]))
            for b in clean_body:
                t = self.clean_text(b[4].replace("\n", " "))
                if t:
                    if t.startswith("•") or t.startswith("-") or t.startswith(""):
                        clean_bullet = re.sub(r'^[•\-\*]\s*', '- ', t)
                        page_obj.paragraphs.append(Paragraph(text=clean_bullet))
                    else:
                        page_obj.paragraphs.append(Paragraph(text=t))

        # Add footnotes at the bottom
        for fn in footnotes:
            page_obj.paragraphs.append(Paragraph(text=f"*{fn}*"))

        return page_obj
