"""
Document Registry
-----------------
Scans all PDFs and generates metadata for each document.
"""
from config import DEFAULT_COMPANY
from pathlib import Path
import hashlib
import json
import re


class DocumentRegistry:

    def __init__(self):

        self.document_folder = Path("data/documents")
        self.metadata_folder = Path("data/metadata")

        self.metadata_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # SHA256 Checksum
    # --------------------------------------------------

    def generate_checksum(self, filepath):

        sha256 = hashlib.sha256()

        with open(filepath, "rb") as file:

            while True:

                chunk = file.read(8192)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # --------------------------------------------------
    # Document Type
    # --------------------------------------------------

    def detect_document_type(self, filename):

        name = filename.lower()

        if "drhp" in name:
            return "DRHP"

        elif "ar" in name:
            return "Annual Report"

        elif "result" in name:
            return "Quarterly Result"

        elif "ppt" in name:
            return "Investor Presentation"

        elif "concall" in name:
            return "Earnings Call Transcript"

        return "Other"

    # --------------------------------------------------
    # Fiscal Year
    # --------------------------------------------------

    def detect_fiscal_year(self, filename):

        match = re.search(r"FY(\d{2})", filename.upper())

        if match:

            year = match.group(1)

            return f"FY20{year}"

        return None

    # --------------------------------------------------
    # Quarter
    # --------------------------------------------------

    def detect_quarter(self, filename):

        match = re.search(r"Q([1-4])", filename.upper())

        if match:

            return f"Q{match.group(1)}"

        return None

    # --------------------------------------------------
    # Reporting Period
    # --------------------------------------------------

    def reporting_period(self, fy, quarter):

        if fy and quarter:
            return f"{quarter} {fy}"

        elif fy:
            return fy

        return None

    # --------------------------------------------------
    # Company Detection
    # --------------------------------------------------

    def company_name(self, filename):

        return (
        DEFAULT_COMPANY["company_id"],
        DEFAULT_COMPANY["company_name"]
    )

    # --------------------------------------------------
    # Source Tier
    # --------------------------------------------------

    def source_tier(self, doc_type):

        tier1 = [
            "Annual Report",
            "Quarterly Result",
            "Investor Presentation",
            "DRHP",
            "Earnings Call Transcript"
        ]

        if doc_type in tier1:
            return 1

        return 2

    # --------------------------------------------------
    # Document ID
    # --------------------------------------------------

    def document_id(self, company_id, fy, quarter, doc_type):

        doc = doc_type.upper().replace(" ", "_")

        if fy and quarter:

            return f"{company_id}_{fy}_{quarter}_{doc}"

        elif fy:

            return f"{company_id}_{fy}_{doc}"

        return f"{company_id}_{doc}"

    # --------------------------------------------------
    # Build Metadata
    # --------------------------------------------------

    def build_metadata(self, filepath):

        filename = filepath.name

        company_id, company_name = self.company_name(filename)

        doc_type = self.detect_document_type(filename)

        fy = self.detect_fiscal_year(filename)

        quarter = self.detect_quarter(filename)

        reporting_period = self.reporting_period(fy, quarter)

        metadata = {

            "doc_id": self.document_id(
                company_id,
                fy,
                quarter,
                doc_type
            ),

            "checksum": self.generate_checksum(filepath),

            "filename": filename,

            "filepath": str(filepath.resolve()),

            "company_id": company_id,

            "company_name": company_name,

            "doc_type": doc_type,

            "source_tier": self.source_tier(doc_type),

            "fiscal_year": fy,

            "quarter": quarter,

            "reporting_period": reporting_period,

            "language": "English",

            "version": "1.0"
        }

        return metadata

    # --------------------------------------------------
    # Process All PDFs
    # --------------------------------------------------

    def process_documents(self):

        pdfs = list(self.document_folder.glob("*.pdf"))

        print(f"\nFound {len(pdfs)} PDF files\n")

        all_metadata = []

        for pdf in pdfs:

            metadata = self.build_metadata(pdf)

            all_metadata.append(metadata)

            output_file = self.metadata_folder / f"{metadata['doc_id']}.json"

            with open(output_file, "w", encoding="utf-8") as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )

            print(f"✓ {metadata['doc_id']}")

        print("\nMetadata generation completed.")

        return all_metadata