"""
Chunk Formatter
---------------

Creates RAG-ready chunks for financial documents.

Supported Formats
-----------------
1. Heading + Paragraph
2. Long Paragraph Split
3. Heading + Table
4. Heading + Figure
5. Heading + KPI Cards
6. Heading + Bullet List
7. Heading + Numbered List
8. Heading + Chart
9. Heading + Mixed Content

Target Chunk Size:
400-500 words
"""

import uuid

from retrieval.chunk_types import ChunkType


class ChunkFormatter:

    def __init__(self, max_words=450):

        self.max_words = max_words

    # =====================================================

    def words(self, text):

        return len(text.split())

    # =====================================================

    def build_chunk(

        self,

        page,

        chunk_type,

        text,

        **extra

    ):

        metadata = dict(page.metadata)

        metadata.update(extra)

        metadata["chunk_type"] = chunk_type

        metadata["page_number"] = page.page_number

        metadata["heading"] = page.title

        metadata["word_count"] = self.words(text)

        metadata["char_count"] = len(text)

        # Low Information Filtering
        raw_words = len(text.split())
        is_data_chunk = chunk_type in {ChunkType.HEADING_TABLE, ChunkType.CHART, ChunkType.KPI_TABLE}
        
        # Skip embedding empty chunks, title-only duplicates, or chunks with <= 15 words (unless data-rich)
        if (not text.strip()) or (text.strip() == page.title.strip()) or (raw_words <= 15 and not is_data_chunk):
            return None

        # Build contextual metadata header to prepend to chunk text
        header = (
            f"Company: {metadata.get('company_name', 'Ather Energy')}\n"
            f"Document Type: {metadata.get('document_type', 'Investor Presentation')}\n"
            f"Reporting Period: {metadata.get('reporting_period', 'Q1 FY2026')}\n"
            f"Page Number: {metadata.get('page_number', page.page_number)}\n"
            f"Topic: {metadata.get('topic', 'general')}\n"
            f"Document Scope: {metadata.get('document_scope', 'current')}\n"
            f"Section Heading: {metadata.get('heading', page.title)}\n\n"
        )
        rich_text = header + text

        return {

            "chunk_id": str(uuid.uuid4()),

            "text": rich_text.strip(),

            "metadata": metadata

        }

    # =====================================================
    # Heading + Paragraph
    # =====================================================

    def paragraph_chunks(self, page):

        chunks = []

        heading = page.title.strip()

        current = heading + "\n\n"

        for para in page.paragraphs:

            text = para.text.strip()

            if not text:

                continue

            if self.words(current + text) > self.max_words:

                chunks.append(

                    self.build_chunk(

                        page,

                        ChunkType.HEADING_PARAGRAPH,

                        current

                    )

                )

                current = heading + "\n\n"

            current += text + "\n\n"

        if current.strip() != heading:

            chunks.append(

                self.build_chunk(

                    page,

                    ChunkType.HEADING_PARAGRAPH,

                    current

                )

            )

        return chunks

    # =====================================================
    # Heading + Table
    # =====================================================

    def table_chunks(self, page):

        chunks = []

        heading = page.title.strip()

        for table in page.tables:

            text = heading + "\n\n"

            if table.title:

                text += table.title + "\n\n"

            for row in table.rows:

                text += " | ".join(row)

                text += "\n"

            chunks.append(

                self.build_chunk(

                    page,

                    ChunkType.HEADING_TABLE,

                    text,

                    table_rows=len(table.rows)

                )

            )

        return chunks

    # =====================================================
    # Heading + Figure
    # =====================================================

    def figure_chunks(self, page):

        chunks = []

        heading = page.title.strip()

        for fig in page.figures:

            text = heading + "\n\n"

            text += f"Figure Type : {fig.figure_type}\n\n"

            if fig.ocr_text:

                text += fig.ocr_text

            chunks.append(

                self.build_chunk(

                    page,

                    ChunkType.HEADING_FIGURE,

                    text,

                    figure_type=fig.figure_type

                )

            )

        return chunks

    # =====================================================
    # KPI Cards
    # =====================================================

    def kpi_chunk(self, page):

        heading = page.title.lower()

        if "financial highlights" not in heading:

            return None

        text = page.title + "\n\n"

        for para in page.paragraphs:

            text += para.text + "\n"

        return self.build_chunk(

            page,

            ChunkType.KPI_TABLE,

            text

        )

    # =====================================================
    # Bullet List
    # =====================================================

    def bullet_chunk(self, page):

        bullets = []

        for para in page.paragraphs:

            t = para.text.strip()

            if t.startswith("•") or t.startswith("-"):

                bullets.append(t)

        if not bullets:

            return None

        text = page.title + "\n\n"

        text += "\n".join(bullets)

        return self.build_chunk(

            page,

            ChunkType.BULLET,

            text

        )

    # =====================================================
    # Numbered List
    # =====================================================

    def numbered_chunk(self, page):

        numbered = []

        for para in page.paragraphs:

            t = para.text.strip()

            if len(t) > 2 and t[0].isdigit():

                numbered.append(t)

        if not numbered:

            return None

        text = page.title + "\n\n"

        text += "\n".join(numbered)

        return self.build_chunk(

            page,

            ChunkType.NUMBERED,

            text

        )

    # =====================================================
    # Chart
    # =====================================================

    def chart_chunks(self, page):

        chunks = []

        for fig in page.figures:

            if fig.figure_type != "financial_chart":

                continue

            text = page.title + "\n\n"

            text += "Chart\n\n"

            text += fig.ocr_text

            chunks.append(

                self.build_chunk(

                    page,

                    ChunkType.CHART,

                    text

                )

            )

        return chunks

    # =====================================================
    # Mixed Chunk
    # =====================================================

    def mixed_chunk(self, page):

        text = page.title + "\n\n"

        for para in page.paragraphs:

            text += para.text + "\n\n"

        for table in page.tables:

            if table.title:

                text += table.title + "\n"

            for row in table.rows:

                text += " | ".join(row)

                text += "\n"

            text += "\n"

        for fig in page.figures:

            if fig.figure_type != "logo_or_icon":

                text += "\n"

                text += fig.ocr_text

                text += "\n"

        return self.build_chunk(

            page,

            ChunkType.MIXED,

            text

        )

    def create_chunks(self, page):

        chunks = []

        chunks.extend(self.paragraph_chunks(page))

        chunks.extend(self.table_chunks(page))

        chunks.extend(self.figure_chunks(page))

        chunks.extend(self.chart_chunks(page))

        kpi = self.kpi_chunk(page)

        if kpi:

            chunks.append(kpi)

        bullet = self.bullet_chunk(page)

        if bullet:

            chunks.append(bullet)

        numbered = self.numbered_chunk(page)

        if numbered:

            chunks.append(numbered)

        chunks.append(self.mixed_chunk(page))

        # Filter out None chunks resulting from low information filtering
        chunks = [c for c in chunks if c is not None]

        # Helper function for calculating the overlap coefficient
        def text_overlap(t1, t2):
            # Strip off the identical metadata header prefix when comparing text
            raw_t1 = t1.split("\n\n", 6)[-1] if "\n\n" in t1 else t1
            raw_t2 = t2.split("\n\n", 6)[-1] if "\n\n" in t2 else t2
            
            w1 = set(raw_t1.lower().split())
            w2 = set(raw_t2.lower().split())
            if not w1 or not w2:
                return 0.0
            intersection = w1.intersection(w2)
            return len(intersection) / min(len(w1), len(w2))

        # Priority scoring: mixed > heading_paragraph > bullet/numbered > table/kpi > figure > chart
        PRIORITY = {
            ChunkType.MIXED: 0,
            ChunkType.HEADING_PARAGRAPH: 1,
            ChunkType.BULLET: 2,
            ChunkType.NUMBERED: 2,
            ChunkType.KPI_TABLE: 3,
            ChunkType.HEADING_TABLE: 3,
            ChunkType.HEADING_FIGURE: 4,
            ChunkType.CHART: 5
        }

        # Sort chunks such that higher priority (smaller priority index) comes first.
        # Tie-break using length (longer first) to maximize rich context preservation.
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                PRIORITY.get(c["metadata"].get("chunk_type"), 99),
                -len(c["text"])
            )
        )
        unique_chunks = []

        for candidate in sorted_chunks:
            is_duplicate = False
            for kept in unique_chunks:
                if text_overlap(candidate["text"], kept["text"]) >= 0.80:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(candidate)

        return unique_chunks