from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store import VectorStore


embedder = EmbeddingGenerator()

store = VectorStore()


query = "products offered by the company"


query_vector = embedder.generate_query_embedding(query)


from config import TOP_K

results = store.search(
    query_vector,
    limit=TOP_K
)


print("\nSEARCH RESULTS\n")

for i, result in enumerate(results):

    print("=" * 80)

    print(f"Result {i + 1}")

    print(f"Similarity Score : {result.score:.4f}")

    print()

    payload = result.payload

    print(payload["text"])

    print()

    print("Metadata:")

    for key, value in payload.items():

        if key != "text":

            print(f"{key}: {value}")

    print()
