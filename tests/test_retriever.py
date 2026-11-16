from retrieval.retriever import Retriever


retriever = Retriever()

query = input("Enter your question: ")

results = retriever.search(query)

print("\n" + "=" * 80)

print("Retrieved Evidence")

print("=" * 80)

for i, evidence in enumerate(results, start=1):

    print(f"\nEvidence {i}")

    print("-" * 60)

    print("Company      :", evidence.company)

    print("Document     :", evidence.document)

    print("Source Type  :", evidence.source_type)

    print("Report Year  :", evidence.report_year)

    print("Page         :", evidence.page)

    print("Chunk ID     :", evidence.chunk_id)

    print("Similarity   :", round(evidence.retrieval_score, 4))

    print("\nText:\n")

    print(evidence.text[:500])

    print("\n")