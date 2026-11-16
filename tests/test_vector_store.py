from pathlib import Path

from ingestion.pipeline import IngestionPipeline

from retrieval.embeddings import EmbeddingGenerator

from retrieval.vector_store import VectorStore

from config import QDRANT_BATCH_SIZE


def main():
    # ------------------------------------------------------
    # Parse PDF
    # ------------------------------------------------------

    pipeline = IngestionPipeline()

    document = pipeline.process_pdf(

        Path("data/documents/Q1FY26 PPT.pdf")

    )

    # ------------------------------------------------------
    # Chunk Formatting
    # ------------------------------------------------------
    print(type(document))
    print(document)
    embedder = EmbeddingGenerator()

    chunks = embedder.create_chunks(document)
    print(len(chunks))
    print(chunks[0])

    print(f"\nCreated {len(chunks)} formatted chunks.\n")

    # ------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------

    embedded = embedder.generate_embeddings(chunks)

    # ------------------------------------------------------
    # Upload
    # ------------------------------------------------------

    store = VectorStore()

    store.recreate_collection(

        len(embedded[0]["embedding"])

    )

    store.upload_embeddings(

        embedded,

        batch_size=QDRANT_BATCH_SIZE

    )

    store.collection_info()


if __name__ == "__main__":
    main()