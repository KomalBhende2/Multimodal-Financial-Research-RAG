"""
Semantic Chunker
----------------

Main chunking pipeline.

Pipeline

Document
      ↓
Chunk Formatter
      ↓
Formatted Chunks
      ↓
Chunk Statistics
"""

from retrieval.chunk_formatter import ChunkFormatter
from retrieval.chunk_statistics import ChunkStatistics


class SemanticChunker:

    def __init__(self):

        self.formatter = ChunkFormatter()

        self.statistics = ChunkStatistics()

    # --------------------------------------------------------

    def create_chunks(self, document):

        chunks = []

        print("\nCreating formatted chunks...\n")

        for page in document.pages:

            page_chunks = self.formatter.create_chunks(page)

            chunks.extend(page_chunks)

        stats = self.statistics.analyze(chunks)

        self.statistics.print_report(stats)

        print(f"\nFinal Chunks : {len(chunks)}\n")

        return chunks