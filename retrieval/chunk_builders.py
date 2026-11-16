"""
Chunk Builders
--------------

Contains helper functions for creating
different RAG chunk formats.
"""

import re


# ----------------------------------------------------------
# KPI Detection
# ----------------------------------------------------------

def extract_kpis(text):

    pattern = re.findall(

        r"([\d,.]+%?|₹?[\d,.]+\s?(?:Mn|Cr|Bn|Lakh|Crore)?)",

        text

    )

    return pattern


# ----------------------------------------------------------
# Bullet List
# ----------------------------------------------------------

def is_bullet(paragraph):

    p = paragraph.strip()

    bullets = (

        "•",

        "-",

        "*",

        "▪",

        "◦"

    )

    return any(

        p.startswith(b)

        for b in bullets

    )


# ----------------------------------------------------------
# Numbered List
# ----------------------------------------------------------

def is_numbered(paragraph):

    return bool(

        re.match(r"^\d+\.", paragraph.strip())

    )


# ----------------------------------------------------------
# Timeline
# ----------------------------------------------------------

def is_timeline(text):

    years = re.findall(

        r"(20\d{2})",

        text

    )

    return len(years) >= 2


# ----------------------------------------------------------
# Financial Chart
# ----------------------------------------------------------

def is_financial_chart(figure):

    return figure.figure_type == "financial_chart"


# ----------------------------------------------------------
# KPI Figure
# ----------------------------------------------------------

def is_kpi_figure(figure):

    if not figure.ocr_text:

        return False

    text = figure.ocr_text.lower()

    words = [

        "revenue",

        "margin",

        "ebitda",

        "income",

        "sales",

        "pat"

    ]

    return any(

        w in text

        for w in words

    )