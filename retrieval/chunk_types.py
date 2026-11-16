"""
Chunk Types
-----------

Defines all supported chunk formats.

These chunk types are attached to every chunk and are useful for:
- Retrieval
- Filtering
- Analytics
- Future reranking
"""


class ChunkType:

    # ==========================
    # Text
    # ==========================

    HEADING_PARAGRAPH = "heading_paragraph"

    LONG_PARAGRAPH = "long_paragraph"

    # ==========================
    # Tables
    # ==========================

    HEADING_TABLE = "heading_table"

    HEADING_TABLE_NOTES = "heading_table_notes"

    FINANCIAL_TABLE = "financial_table"

    KPI_TABLE = "kpi_table"

    # ==========================
    # Figures
    # ==========================

    HEADING_FIGURE = "heading_figure"

    HEADING_CHART = "heading_chart"

    INFOGRAPHIC = "infographic"

    FLOWCHART = "flowchart"

    TIMELINE = "timeline"

    # ==========================
    # Lists
    # ==========================

    BULLET = "bullet_list"

    NUMBERED = "numbered_list"

    FAQ = "faq"

    # ==========================
    # Business
    # ==========================

    CEO_COMMENTARY = "ceo_commentary"

    BOARD_COMMENTARY = "board_commentary"

    PRODUCT = "product"

    PEER_COMPARISON = "peer_comparison"

    INDUSTRY_DATA = "industry_data"

    RISK = "risk"

    ESG = "esg"

    # ==========================
    # Financial Statements
    # ==========================

    BALANCE_SHEET = "balance_sheet"

    INCOME_STATEMENT = "income_statement"

    CASH_FLOW = "cash_flow"

    SHAREHOLDING = "shareholding"

    CHART = "chart"

    # ==========================
    # Mixed
    # ==========================

    MIXED = "mixed"