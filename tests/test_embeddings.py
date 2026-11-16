from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import Chunker
from retrieval.embeddings import EmbeddingGenerator

loader = PDFLoader()

documents = loader.load_documents()

chunker = Chunker()

chunks = chunker.create_chunks(documents)

embedder = EmbeddingGenerator()

embedded_chunks = embedder.generate_embeddings(chunks)

print("\nTotal Embedded Chunks:")

print(len(embedded_chunks))

print("\nMetadata:\n")

print(embedded_chunks[0]["metadata"])

print("\nEmbedding Dimension:\n")

print(len(embedded_chunks[0]["embedding"]))