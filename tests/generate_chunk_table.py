from pathlib import Path
import csv
from ingestion.pipeline import IngestionPipeline
from retrieval.chunk_formatter import ChunkFormatter
from retrieval.chunk_statistics import ChunkStatistics

def main():
    # 1. Parse PDF
    pipeline = IngestionPipeline()
    document = pipeline.process_pdf(
        Path("data/documents/Q1FY26 PPT.pdf")
    )

    # 2. Chunk Formatting
    formatter = ChunkFormatter()
    stats_calculator = ChunkStatistics()

    chunks = []
    for page in document.pages:
        page_chunks = formatter.create_chunks(page)
        chunks.extend(page_chunks)

    # 3. Calculate statistics
    stats = stats_calculator.calculate(chunks)

    # 4. Generate CSV file
    csv_path = Path("data/chunk_details.csv")
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Chunk Index", "Page Number", "Chunk Type", "Word Count", "Char Count", "Heading", "Text Snippet"])
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            snippet = chunk["text"].replace("\n", " ")[:60] + "..."
            writer.writerow([
                idx,
                meta.get("page_number", ""),
                meta.get("chunk_type", ""),
                meta.get("word_count", 0),
                meta.get("char_count", 0),
                meta.get("heading", ""),
                snippet
            ])

    # 5. Generate Markdown Artifact
    artifact_dir = Path(r"C:\Users\komal\.gemini\antigravity\brain\085051cc-f5a3-4105-a1b5-c351b11bd1b1")
    artifact_dir.mkdir(exist_ok=True, parents=True)
    md_path = artifact_dir / "chunk_details_report.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Chunk Details Report\n\n")
        f.write("This report provides a detailed descriptive statistics table of all 91 chunks generated from `Q1FY26 PPT.pdf`.\n\n")
        
        f.write("## Chunk Summary Metrics\n\n")
        f.write(f"- **Total Chunks**: {stats['total_chunks']}\n")
        f.write(f"- **Average Words per Chunk**: {stats['average_words']}\n")
        f.write(f"- **Minimum Words**: {stats['minimum_words']}\n")
        f.write(f"- **Maximum Words**: {stats['maximum_words']}\n")
        f.write(f"- **Average Characters**: {stats['average_characters']}\n\n")
        
        f.write("## Detailed Chunk Table\n\n")
        f.write("| Chunk # | Page | Chunk Type | Words | Characters | Heading | Snippet |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            snippet = chunk["text"].replace("\n", " ").replace("|", "\\|")[:50]
            if len(chunk["text"]) > 50:
                snippet += "..."
            heading = meta.get("heading", "").replace("\n", " ").replace("|", "\\|")
            f.write(f"| {idx} | {meta.get('page_number', '')} | `{meta.get('chunk_type', '')}` | {meta.get('word_count', 0)} | {meta.get('char_count', 0)} | {heading} | {snippet} |\n")

    print(f"Successfully generated chunk CSV: {csv_path}")
    print(f"Successfully generated Markdown report: {md_path}")

if __name__ == "__main__":
    main()
