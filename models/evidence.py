"""
Evidence Object
---------------
Represents one piece of retrieved evidence.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Evidence:

    # Company Information
    company: str

    # Document Information
    document: str
    source_type: str
    report_year: int

    # Location Information
    page: int
    chunk_id: int

    # Content
    text: str

    # Embedding Vector
    embedding: List[float]

    # Retrieval Score
    retrieval_score: Optional[float] = None