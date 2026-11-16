"""
Embedding Generator
-------------------

Pipeline

Document
    ↓
Chunk Formatter
    ↓
Formatted Chunks
    ↓
Embeddings
"""

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

from retrieval.chunk_formatter import ChunkFormatter
from retrieval.chunk_statistics import ChunkStatistics


class EmbeddingGenerator:

    def __init__(self):

        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True
        )

        self.formatter = ChunkFormatter()

        self.stats = ChunkStatistics()

        print("Embedding Model Loaded.\n")

    # ----------------------------------------------------
    # Build chunks from parsed document
    # ----------------------------------------------------

    def create_chunks(self, document):

        chunks = []

        for page in document.pages:

            page_chunks = self.formatter.create_chunks(page)

            chunks.extend(page_chunks)

            

        stats = self.stats.calculate(chunks)
        self.stats.print_report(stats)

        return chunks

    # ----------------------------------------------------
    # Embedding Generation
    # ----------------------------------------------------

    def generate_embeddings(self, chunks):

        print("\nGenerating Embeddings...\n")

        texts = [

            chunk["text"]

            for chunk in chunks

        ]

        try:
            print("Starting multi-process pool for parallel encoding...")
            pool = self.model.start_multi_process_pool()
            vectors = self.model.encode_multi_process(
                texts,
                pool,
                normalize_embeddings=True
            )
            self.model.stop_multi_process_pool(pool)
            print("Parallel encoding complete.")
        except Exception as e:
            print(f"Fallback to single-process encoding due to: {e}")
            vectors = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )

        embedded = []

        for chunk, vector in zip(chunks, vectors):

            embedded.append(

                {

                    "text": chunk["text"],

                    "embedding": vector.tolist(),

                    "metadata": chunk["metadata"]

                }

            )

        return embedded

    # ----------------------------------------------------
    # Query Embedding
    # ----------------------------------------------------

    def generate_query_embedding(self, query):

        vector = self.model.encode(

            query,

            normalize_embeddings=True

        )

        return vector.tolist()