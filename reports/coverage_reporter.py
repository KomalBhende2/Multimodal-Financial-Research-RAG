import os
from datetime import datetime
from typing import List, Dict, Set, Any
from models.evidence import Evidence

class CoverageReporter:
    """
    CoverageReporter
    ----------------
    Analyzes which query themes were satisfied by the selected evidence chunks,
    calculates coverage statistics, and exports Markdown reports.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, query: str, requested_themes: List[str], selected_evidence: List[Evidence], retriever_instance: Any = None) -> str:
        """
        Generates a detailed Markdown report showing coverage score, mapped evidence,
        and outstanding theme gaps.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate theme mappings
        theme_to_chunks = {theme: [] for theme in requested_themes}
        uncategorized_chunks = []
        
        for idx, ev in enumerate(selected_evidence, 1):
            # Dynamically classify themes if retriever instance is provided
            if retriever_instance and hasattr(retriever_instance, "classify_chunk_themes"):
                chunk_themes = retriever_instance.classify_chunk_themes(ev.text, ev.topic + " " + ev.document)
            else:
                # Basic fallback classifier keyword matching
                chunk_themes = []
                t = ev.text.lower() + " " + ev.document.lower()
                if "history" in t or "founded" in t: chunk_themes.append("History")
                if "milestones" in t or "timeline" in t: chunk_themes.append("Timeline")
                if "manufacturing" in t or "facilities" in t: chunk_themes.append("Manufacturing")
                if "certif" in t: chunk_themes.append("Certifications")
                if "global" in t or "export" in t: chunk_themes.append("Global Presence")
                if "oem" in t or "customer" in t: chunk_themes.append("Customers")
                if "product" in t or "portfolio" in t: chunk_themes.append("Products")
                if "strategy" in t: chunk_themes.append("Strategy")
                if "overview" in t: chunk_themes.append("Company Overview")
                if not chunk_themes: chunk_themes.append("General")

            matched_any = False
            for theme in requested_themes:
                if theme in chunk_themes:
                    theme_to_chunks[theme].append((idx, ev))
                    matched_any = True
            
            if not matched_any:
                uncategorized_chunks.append((idx, ev))

        # Calculate coverage stats
        total_requested = len(requested_themes)
        covered_themes = [theme for theme in requested_themes if len(theme_to_chunks[theme]) > 0]
        covered_count = len(covered_themes)
        coverage_pct = (covered_count / total_requested * 100.0) if total_requested > 0 else 100.0
        gaps = [theme for theme in requested_themes if len(theme_to_chunks[theme]) == 0]

        # Construct report content
        lines = []
        lines.append("# RAG Retrieval Coverage Report")
        lines.append(f"**Timestamp:** {timestamp}  ")
        lines.append(f"**Query:** `{query}`\n")
        
        lines.append("## Coverage Metrics")
        lines.append(f"- **Requested Aspects/Themes:** {total_requested}")
        lines.append(f"- **Covered Themes:** {covered_count} / {total_requested}")
        lines.append(f"- **Coverage Score:** {coverage_pct:.1f}%")
        lines.append(f"- **Outstanding Gaps:** {len(gaps)}")
        lines.append("")
        
        # Display progress bar
        bar_len = 20
        filled_len = int(round(bar_len * covered_count / total_requested)) if total_requested > 0 else bar_len
        bar = "█" * filled_len + "░" * (bar_len - filled_len)
        lines.append(f"**Progress:** `[{bar}]` {coverage_pct:.1f}%\n")

        # Display theme breakdown table
        lines.append("## Aspect Breakdown")
        lines.append("| Aspect / Theme | Status | Mapped Results |")
        lines.append("| :--- | :--- | :--- |")
        for theme in requested_themes:
            mapped_items = theme_to_chunks[theme]
            if mapped_items:
                status = "✅ Covered"
                refs = ", ".join(f"Result {idx} (Page {ev.page})" for idx, ev in mapped_items)
            else:
                status = "❌ Gap / Uncovered"
                refs = "None"
            lines.append(f"| {theme} | {status} | {refs} |")
        lines.append("")

        # Display details of selected chunks
        lines.append("## Selected Evidence Details")
        for idx, ev in enumerate(selected_evidence, 1):
            lines.append(f"### Result {idx}: Page {ev.page} ({ev.document})")
            lines.append(f"- **Company:** {ev.company}")
            lines.append(f"- **Source Type:** {ev.source_type}")
            lines.append(f"- **Retrieval/Rerank Score:** {ev.retrieval_score:.4f}")
            lines.append(f"- **Snippet:**")
            snippet = ev.text.replace("\n", " ").strip()[:200] + "..."
            lines.append(f"  > {snippet}")
            lines.append("")

        # Display outstanding gap analysis
        if gaps:
            lines.append("## Gap Analysis")
            lines.append("The following requested query aspects could not be satisfied from the candidate evidence pool:")
            for gap in gaps:
                lines.append(f"- **{gap}**: No relevant non-redundant chunk matching this topic was found above the similarity threshold.")
            lines.append("")

        return "\n".join(lines)

    def save_report_to_file(self, query: str, report_content: str) -> str:
        """
        Saves the report content to the reports/ directory.
        """
        clean_query = "".join(c if c.isalnum() else "_" for c in query[:30]).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coverage_report_{clean_query}_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"Coverage Report saved to: {filepath}")
        return filepath

# Fallback type check support
from typing import Any
