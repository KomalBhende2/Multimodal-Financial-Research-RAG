import re
from typing import List, Dict, Any
from models.evidence import Evidence

class EvidenceValidator:
    """
    EvidenceValidator
    -----------------
    Validates retrieved evidence chunks against structural rules, company filters,
    and query context to ensure relevance and validity.
    """

    def __init__(self):
        # List of supported companies and their variations for alignment checks
        self.company_mappings = {
            "bajaj_finance": ["bajaj"],
            "pine_labs": ["pine"],
            "hscl": ["hscl", "himadri"],
            "azad_engineering": ["azad"],
            "bank_of_maharashtra": ["maharashtra"],
            "ratnaveer": ["ratnaveer", "ratanveer"],
            "ather": ["ather"]
        }

    def detect_company_id(self, text: str) -> str:
        q = text.lower()
        for company_id, keywords in self.company_mappings.items():
            if any(k in q for k in keywords):
                return company_id
        return None

    def validate_chunk(self, query: str, ev: Evidence) -> Dict[str, Any]:
        """
        Validates a single evidence chunk.
        """
        errors = []
        
        # 1. Check content length and validity
        if not ev.text or not ev.text.strip():
            errors.append("Evidence content is empty.")
        elif len(ev.text.strip()) < 50:
            errors.append(f"Evidence content is too short ({len(ev.text.strip())} chars; min 50).")

        # 2. Check location validity
        if ev.page < 0:
            errors.append(f"Invalid page number ({ev.page}). Page numbers must be non-negative.")

        # 3. Check score validity
        if ev.retrieval_score is not None:
            if ev.retrieval_score < 0.0 or ev.retrieval_score > 1.0:
                errors.append(f"Retrieval score ({ev.retrieval_score:.4f}) is out of [0.0, 1.0] bounds.")

        # 4. Check company alignment (if query explicitly names a company)
        query_company = self.detect_company_id(query)
        if query_company:
            chunk_company = self.detect_company_id(ev.company + " " + ev.text)
            # If the chunk clearly mentions another company entirely, report a mismatch
            if chunk_company and chunk_company != query_company:
                errors.append(f"Company mismatch: query targets '{query_company}' but chunk references '{chunk_company}'.")

        # 5. Check semantic term coverage (relevance validation)
        # Extract meaningful terms from query to check text intersection
        query_words = set(re.findall(r"\w+", query.lower()))
        stop_words = {"what", "is", "are", "the", "and", "or", "for", "in", "of", "to", "at", "by", "on", "with", "about", "company"}
        keywords = query_words - stop_words
        
        # We expect at least one query keyword (or company name token) in the chunk text
        chunk_words = set(re.findall(r"\w+", ev.text.lower()) + re.findall(r"\w+", ev.company.lower()))
        if keywords and not (keywords & chunk_words):
            errors.append("Low relevance: no core query concepts/keywords found in the chunk content.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "details": {
                "page": ev.page,
                "company": ev.company,
                "document": ev.document,
                "score": ev.retrieval_score,
                "snippet": ev.text[:100] + "..." if ev.text else ""
            }
        }

    def validate_all(self, query: str, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """
        Validates all evidence chunks and returns individual reports.
        """
        return [self.validate_chunk(query, ev) for ev in evidence_list]

    def filter_valid(self, query: str, evidence_list: List[Evidence]) -> List[Evidence]:
        """
        Validates and filters a list of Evidence, returning only valid chunks.
        """
        valid_chunks = []
        for ev in evidence_list:
            report = self.validate_chunk(query, ev)
            if report["is_valid"]:
                valid_chunks.append(ev)
            else:
                print(f"Validator Warning: Page {ev.page} ({ev.document}) rejected. Reasons: {report['errors']}")
        return valid_chunks
