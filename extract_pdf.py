#!/usr/bin/env python
r"""
Extract text, tables, and figures from PDF and save to structured Markdown file

Usage:
    python extract_pdf.py <pdf_path> [output_path]

Example:
    python extract_pdf.py "data/documents/Q1FY26 PPT.pdf"
    python extract_pdf.py "data/documents/Q1FY26 PPT.pdf" "extracted_output/result.md"
"""

import sys
from pathlib import Path
from ingestion.pipeline import IngestionPipeline


def extract_pdf(pdf_path, output_path=None):
    """Extract text from PDF and save to structured Markdown file"""
    
    pdf_path = Path(pdf_path)
    
    # Validate PDF exists
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return False
    
    print(f"\n[STARTING] Processing: {pdf_path.name}")
    print("=" * 70)
    
    # Run pipeline
    pipeline = IngestionPipeline()
    document = pipeline.process_pdf(pdf_path)
    
    # Determine output path
    if output_path is None:
        output_path = Path("extracted_output") / f"{pdf_path.stem}.md"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build markdown directly to destination
    saved_file = pipeline.builder.build(document, output_folder=output_path.parent)
    
    # If a specific filename was requested other than the default stem.md
    if saved_file != output_path and output_path.suffix == ".md":
        import shutil
        shutil.copy(saved_file, output_path)

    print(f"[SUCCESS] Extraction complete!")
    print(f"[OUTPUT] Saved to: {output_path.resolve()}")
    
    return True


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = extract_pdf(pdf_path, output_path)
    sys.exit(0 if success else 1)
