"""
Metadata Enricher
-----------------

Production Version (Generic & Dynamic)

Features:
✓ Dynamically infers Company Name from document text and filename
✓ Dynamically detects Document Type (Investor Presentation, Annual Report, etc.)
✓ Dynamically extracts Fiscal Year and Quarter
✓ Dynamic Entity and Keyword extraction across any industry
✓ Generic, layout-based Financial and Scope classification
✓ Zero hardcoded company-specific strings
"""

import re
from pathlib import Path
from collections import Counter


class MetadataEnricher:

    def _infer_company_name(self, filename: str, first_pages_text: str) -> str:
        """Dynamically detect company name from text or filename without hardcoding."""
        corporate_patterns = [
            r'For\s+([A-Z][A-Za-z0-9&.,\s]{2,60}?(?:Limited|Ltd|Corporation|Corp|Inc|Enterprises|Industries))\b',
            r'\b([A-Z][A-Za-z0-9&.,\s]{2,60}?(?:Limited|Ltd|Private Limited|Pvt Ltd|Enterprise Limited|Enterprises Limited))\b'
        ]
        
        for pattern in corporate_patterns:
            for match in re.finditer(pattern, first_pages_text, flags=re.IGNORECASE):
                candidate = match.group(1).strip()
                candidate = re.split(r'[\r\n]', candidate)[0].strip()
                # Strip leading salutations / recipient prefixes
                candidate = re.sub(r'^(?:The\s+Secretary,?\s*|To,?\s*)', '', candidate, flags=re.IGNORECASE).strip()
                # Exclude stock exchanges and regulatory bodies
                if any(ex in candidate.lower() for ex in ['stock exchange', 'bse', 'nse', 'compliance', 'the secretary']):
                    continue
                if 3 < len(candidate) < 70:
                    return candidate.title()

        # 2. Extract from filename stem
        stem = Path(filename).stem
        cleaned_stem = re.sub(r'[\-_]?(?:Q[1-4]|FY\d{2,4}|PPT|presentation|report|results|final|v\d+)', '', stem, flags=re.IGNORECASE)
        cleaned_stem = re.sub(r'[_\-]+', ' ', cleaned_stem).strip()
        
        if cleaned_stem and len(cleaned_stem) > 2:
            return cleaned_stem.title()

        return "Unknown Company"

    def _infer_document_type(self, text: str, filename: str) -> str:
        """Dynamically infer document type from content and filename."""
        combined = f"{filename} {text[:3000]}".lower()

        if any(k in combined for k in ["investor presentation", "earnings presentation", "analyst presentation"]):
            return "Investor Presentation"
        if any(k in combined for k in ["annual report", "integrated report"]):
            return "Annual Report"
        if any(k in combined for k in ["quarterly results", "financial results", "earnings release"]):
            return "Financial Results"
        if any(k in combined for k in ["prospectus", "red herring", "drhp", "rhp"]):
            return "Prospectus / RHP"
        if any(k in combined for k in ["earnings call transcript", "conference call transcript"]):
            return "Earnings Call Transcript"

        return "Corporate Presentation"

    def _infer_fiscal_period(self, text: str, filename: str):
        """Extract Quarter and Fiscal Year dynamically."""
        # 1. Search content first for explicit period like Q1 FY27 or Q1FY26
        explicit_match = re.search(r'\b(Q[1-4])\s*(?:FY|Fiscal)?\s*(\d{2,4})\b', text[:3000], re.IGNORECASE)
        if explicit_match:
            quarter = explicit_match.group(1).upper()
            fy_num = explicit_match.group(2)
            fiscal_year = f"FY{fy_num}" if not fy_num.startswith("FY") else fy_num
            return quarter, fiscal_year

        combined = f"{filename} {text[:2000]}"
        
        q_match = re.search(r'\b(Q[1-4])\b', combined, re.IGNORECASE)
        quarter = q_match.group(1).upper() if q_match else ""

        fy_match = re.search(r'\b(?:FY\s*[-_]?\s*(\d{2,4})|(20\d{2}))\b', combined, re.IGNORECASE)
        if fy_match:
            fy_val = fy_match.group(1) or fy_match.group(2)
            fiscal_year = f"FY{fy_val}" if not fy_val.startswith("FY") else fy_val
        else:
            fiscal_year = ""

        return quarter, fiscal_year

    def enrich(self, document):

        filename = Path(document.filename).stem

        # Combine text of first 3 pages for high-level metadata inference
        first_pages_text = " ".join(
            " ".join(p.text for p in page.paragraphs)
            for page in document.pages[:3]
        )

        # -------------------------------------------------
        # Document Metadata (Dynamic)
        # -------------------------------------------------
        if not getattr(document, "company_name", ""):
            document.company_name = self._infer_company_name(document.filename, first_pages_text)

        quarter, fiscal_year = self._infer_fiscal_period(first_pages_text, document.filename)
        if not getattr(document, "quarter", ""):
            document.quarter = quarter
        if not getattr(document, "fiscal_year", ""):
            document.fiscal_year = fiscal_year

        if not getattr(document, "document_type", ""):
            document.document_type = self._infer_document_type(first_pages_text, document.filename)

        document.metadata = {
            "filename": filename,
            "company": document.company_name,
            "document_type": document.document_type,
            "fiscal_year": document.fiscal_year,
            "quarter": document.quarter,
            "total_pages": len(document.pages)
        }

        # -------------------------------------------------
        # Universal Financial Glossary
        # -------------------------------------------------
        financial_words = {
            "revenue", "income", "sales", "profit", "ebitda", "ebit",
            "margin", "pat", "pbt", "cash", "expenses", "volume",
            "units", "market share", "cogs", "capex", "opex", "debt",
            "equity", "turnover", "aov", "sssg", "growth", "cagr",
            "gross profit", "contribution margin", "assets", "liabilities"
        }

        # Generic stop words to exclude from dynamic entity extraction
        entity_stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "our", "are",
            "page", "slide", "table", "total", "year", "quarter", "month",
            "presentation", "company", "overview", "report", "growth", "india",
            "statement", "highlights", "source", "notes", "performance", "ended",
            "annexure", "contents", "disclaimer", "limited", "ltd", "private"
        }

        # Collect entities dynamically across pages
        for page in document.pages:

            paragraph_text = " ".join(p.text for p in page.paragraphs)
            figure_text = " ".join(f.ocr_text for f in page.figures if f.ocr_text)
            full_text = f"{paragraph_text} {figure_text}".strip()
            lower = full_text.lower()

            numbers = re.findall(r"\d+(?:\.\d+)?", full_text)

            financial_metrics = sorted({
                word for word in financial_words
                if re.search(rf"\b{re.escape(word)}\b", lower)
            })

            # Dynamic entity detection: extract capitalized terms (2+ words or distinct proper nouns)
            raw_entities = re.findall(r'\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b', full_text)
            clean_entities = []
            for ent in raw_entities:
                ent_clean = ent.strip()
                if len(ent_clean) > 2 and ent_clean.lower() not in entity_stopwords:
                    clean_entities.append(ent_clean)

            # Keep top distinct entities on this page
            entities = [item[0] for item in Counter(clean_entities).most_common(10)]

            # ------------------------------------------
            # Chart & Diagram Detection
            # ------------------------------------------
            has_chart = any(
                fig.figure_type in {
                    "financial_chart", "chart", "bar_chart", "line_chart", "pie_chart"
                }
                for fig in page.figures
            )

            has_diagram = any(
                fig.figure_type in {"diagram", "workflow", "process"}
                for fig in page.figures
            )

            has_image = len(page.figures) > 0

            # Dynamic Financials Detection
            contains_fin = (
                len(financial_metrics) >= 2
                or len(page.tables) > 0
                or any(term in lower for term in ["profit", "loss", "balance sheet", "p&l", "databook", "ebitda", "inr mn", "crore"])
            )

            # Dynamic Document Scope
            future_keywords = {"ahead", "looking ahead", "upcoming", "roadmap", "future", "guidance", "target", "outlook", "pipeline"}
            hist_keywords = {"yoy", "qoq", "historical", "prior year", "prior period", "cagr", "trajectory"}

            has_future_words = any(fw in lower for fw in future_keywords) or any(fw in page.title.lower() for fw in future_keywords)
            has_hist_words = any(hw in lower for hw in hist_keywords)

            if has_future_words and not has_hist_words:
                doc_scope = "forward_looking"
            elif has_hist_words:
                doc_scope = "historical"
            else:
                doc_scope = "current"

            # Dynamic Topic Classification
            title_lower = page.title.lower()
            if any(term in title_lower for term in ["financial", "ebitda", "margin", "p&l", "revenue", "income", "databook", "balance sheet"]):
                topic = "financials"
            elif any(term in title_lower for term in ["board", "management", "director", "leadership", "executive", "governance"]):
                topic = "governance"
            elif any(term in title_lower for term in ["product", "portfolio", "offering", "design", "collection", "materials", "category"]):
                topic = "product"
            elif any(term in title_lower for term in ["manufacturing", "factory", "plant", "capacity", "operations", "production"]):
                topic = "operations"
            elif any(term in title_lower for term in ["strategy", "omnichannel", "expansion", "network", "store", "penetration", "footprint"]):
                topic = "strategy"
            elif any(term in title_lower for term in ["technology", "tech stack", "digital", "architecture", "software", "in-house"]):
                topic = "technology"
            elif doc_scope == "forward_looking":
                topic = "guidance"
            else:
                topic = "general"

            # Page Metadata Assignment
            page.metadata = {
                "page_number": page.page_number,
                "section": page.title,
                "subsection": page.subsection,
                "heading": page.title,
                "parent_heading": page.parent_heading,
                "chunk_type": "page",
                "has_table": len(page.tables) > 0,
                "has_chart": has_chart,
                "has_diagram": has_diagram,
                "has_image": has_image,
                "has_numbers": len(numbers) > 0,
                "entities": entities,
                "financial_metrics": financial_metrics,
                "company_name": document.company_name,
                "document_type": document.document_type,
                "reporting_period": f"{document.quarter} {document.fiscal_year}".strip(),
                "contains_financials": contains_fin,
                "contains_tables": len(page.tables) > 0,
                "contains_figures": len(page.figures) > 0,
                "contains_charts": has_chart,
                "document_scope": doc_scope,
                "topic": topic
            }

        return document
