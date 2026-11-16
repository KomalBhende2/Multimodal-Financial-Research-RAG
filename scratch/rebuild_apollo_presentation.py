import sys
from pathlib import Path
import re

# Ensure stdout handles utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Read existing file to preserve verified narrative and paragraphs
old_path = Path("extracted_output/investor_presentation_Announcement_under_Regulation_30_LODR_Investor_Presentation_2026-08-12.md")
with open(old_path, "r", encoding="utf-8") as f:
    old_content = f.read()

# Pages to sanitize and reconstruct
# 1. Update front-matter
front_matter = """# Document Extraction: Apollo Hospitals Enterprise Limited

## Document Overview
- **Company**: Apollo Hospitals Enterprise Limited
- **Document Type**: Investor Presentation
- **Reporting Period**: Q1 FY27 (Three months ended June 30, 2026)
- **Total Pages**: 36
- **Source File**: investor_presentation_Announcement_under_Regulation_30_LODR_Investor_Presentation_2026-08-12.pdf

---
"""

# Let's split old content by page dividers
pages_raw = old_content.split("\n---\n")

print(f"Total raw sections: {len(pages_raw)}")
