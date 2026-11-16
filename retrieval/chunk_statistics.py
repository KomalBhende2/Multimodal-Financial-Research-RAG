"""
Chunk Statistics
----------------

Calculates descriptive statistics for generated chunks.

Outputs:

- Total Chunks
- Average Words
- Minimum Words
- Maximum Words
- Average Characters
- Chunk Type Distribution
"""

from collections import Counter
from statistics import mean


class ChunkStatistics:

    def __init__(self):
        pass

    # -------------------------------------------------------

    def calculate(self, chunks):

        if not chunks:

            return {}

        word_counts = []
        char_counts = []

        counter = Counter()

        for chunk in chunks:

            text = chunk["text"]

            words = len(text.split())

            chars = len(text)

            word_counts.append(words)

            char_counts.append(chars)

            chunk_type = chunk["metadata"].get(
                "chunk_type",
                "unknown"
            )

            counter[chunk_type] += 1

        stats = {

            "total_chunks": len(chunks),

            "average_words": round(mean(word_counts), 2),

            "minimum_words": min(word_counts),

            "maximum_words": max(word_counts),

            "average_characters": round(
                mean(char_counts),
                2
            ),

            "chunk_distribution": dict(counter)

        }

        return stats

    # -------------------------------------------------------

    def print_report(self, stats):

        print()

        print("=" * 70)

        print("CHUNK STATISTICS")

        print("=" * 70)

        print(f"Total Chunks       : {stats['total_chunks']}")

        print(f"Average Words      : {stats['average_words']}")

        print(f"Minimum Words      : {stats['minimum_words']}")

        print(f"Maximum Words      : {stats['maximum_words']}")

        print(f"Average Characters : {stats['average_characters']}")

        print()

        print("Chunk Distribution")

        print("-" * 70)

        for name, count in stats["chunk_distribution"].items():

            print(f"{name:<30} {count}")

        print("=" * 70)