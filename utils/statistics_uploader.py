import sys
from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store import VectorStore

class StatisticsUploader:

    def __init__(self):
        # Initialize the project's EmbeddingGenerator and VectorStore
        self.embedder = EmbeddingGenerator()
        self.store = VectorStore()

    def upload_statistics(self):
        # 1. Define the descriptive statistics dictionary
        statistics = {
            "total_chunks": 91,
            "total_words": 5860,
            "total_characters": 35518,
            "mean_words": 64.3956,
            "median_words": 41.0,
            "mode_words": 20,
            "minimum_words": 0,
            "maximum_words": 402,
            "range_words": 402,
            "q1_words": 23.0,
            "q2_words": 41.0,
            "q3_words": 96.5,
            "iqr_words": 73.5,
            "variance": 3786.7306,
            "standard_deviation": 61.5364,
            "coefficient_of_variation": 0.9556,
            "skewness": 2.4142,
            "kurtosis": 9.1843
        }

        # 2. Define the readable summary text
        summary_text = (
            "Chunk Statistics:\n"
            "Total Chunks: 91\n"
            "Total Words: 5860\n"
            "Mean Chunk Size: 64.39 words\n"
            "Median: 41\n"
            "Mode: 20\n"
            "Minimum: 0\n"
            "Maximum: 402\n"
            "Q1: 23\n"
            "Q2: 41\n"
            "Q3: 96.5\n"
            "IQR: 73.5\n"
            "Variance: 3786.73\n"
            "Standard Deviation: 61.54\n"
            "Coefficient of Variation: 0.9556\n"
            "Skewness: 2.4142\n"
            "Kurtosis: 9.1843"
        )

        # 3. Generate the embedding for the summary text
        print("Generating embedding for the descriptive statistics document...")
        embedding = self.embedder.generate_query_embedding(summary_text)

        # 4. Construct payload metadata
        metadata = {
            "document_type": "chunk_statistics",
            "company_id": "ATHER",
            "company_name": "Ather Energy Limited",
            "statistics_version": "v1",
            "generated_from": "chunk_formatter",
            "source": "descriptive_statistics",
        }
        # Add the raw numerical statistics to the metadata
        metadata.update(statistics)

        # 5. Ensure the collection exists in Qdrant
        embedding_size = len(embedding)
        print(f"Ensuring Qdrant collection exists (dimension={embedding_size})...")
        self.store.create_collection(embedding_size=embedding_size)

        # 6. Format the embedded document for VectorStore upload
        embedded_doc = {
            "text": summary_text,
            "embedding": embedding,
            "metadata": metadata
        }

        # 7. Upload to Qdrant
        print("Uploading the statistics document to Qdrant...")
        self.store.upload_embeddings([embedded_doc])
        print("Statistics document uploaded successfully.")

if __name__ == "__main__":
    uploader = StatisticsUploader()
    uploader.upload_statistics()
