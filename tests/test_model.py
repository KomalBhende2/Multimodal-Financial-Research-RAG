from sentence_transformers import SentenceTransformer

print("Loading...")

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

print("Loaded Successfully")