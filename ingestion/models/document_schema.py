"""
Production Document Schema
--------------------------

Common schema shared by every extractor.

Everything in the pipeline works with these classes,
NOT with Docling objects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ===========================================================
# Paragraph
# ===========================================================

@dataclass
class Paragraph:

    text: str

    heading_level: int = 0

    source: str = "docling"

    confidence: float = 1.0


# ===========================================================
# Table
# ===========================================================

@dataclass
class Table:

    title: str = ""

    rows: List[List[str]] = field(default_factory=list)

    source: str = "docling"
    
    confidence: float = 1.0

    page_number: Optional[int] = None

    bbox: Optional[List[float]] = None


# ===========================================================
# Figure
# ===========================================================

@dataclass
class Figure:
    page_number: int = 0
    
    image_path: str = ""

    caption: str = ""

    ocr_text: str = ""

    source: str = "docling"

    figure_type: str = "unknown"

    width: int = 0

    height: int = 0


# ===========================================================
# Page
# ===========================================================

@dataclass
class Page:

    page_number: int

    title: str = ""

    paragraphs: List[Paragraph] = field(default_factory=list)
    parent_heading: str = ""
    subsection: str = ""

    tables: List[Table] = field(default_factory=list)

    figures: List[Figure] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)


# ===========================================================
# Document
# ===========================================================

@dataclass
class Document:

    filename: str

    company_name: str = ""

    document_type: str = ""

    fiscal_year: str = ""

    quarter: str = ""

    pages: List[Page] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)

    @property
    def company(self) -> str:
        return self.company_name

    @company.setter
    def company(self, value: str):
        self.company_name = value
