from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import Chunker

loader = PDFLoader()
documents = loader.load_documents()

chunker = Chunker()

chunks = chunker.create_chunks(documents)

print("\nTotal Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0]["metadata"])

print("\n")

print(chunks[0]["text"][:500])