from pathlib import Path
from ingestion.pipeline import IngestionPipeline
from retrieval.chunk_formatter import ChunkFormatter
from retrieval.chunk_statistics import ChunkStatistics

def main():
    # 1. Parse PDF batch-by-batch
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

    # 3. Calculate and print statistics report
    stats = stats_calculator.calculate(chunks)
    stats_calculator.print_report(stats)

if __name__ == "__main__":
    main()
