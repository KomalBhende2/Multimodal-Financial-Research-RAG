from uuid import uuid4

from tqdm import tqdm

from qdrant_client import QdrantClient

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from config import (
    COLLECTION_NAME,
    QDRANT_BATCH_SIZE,
    QDRANT_PATH,
)


class VectorStore:

    def __init__(self):

        self.collection_name = COLLECTION_NAME

        self.client = QdrantClient(
            path=QDRANT_PATH
        )

    # =====================================================
    # Create Collection
    # =====================================================

    def create_collection(self, embedding_size):

        collections = [

            c.name

            for c in self.client.get_collections().collections

        ]

        if self.collection_name in collections:

            print("Collection Already Exists.")

            return

        self.client.create_collection(

            collection_name=self.collection_name,

            vectors_config=VectorParams(

                size=embedding_size,

                distance=Distance.COSINE

            )

        )

        print("Collection Created.")

    # =====================================================
    # Recreate Collection
    # =====================================================

    def recreate_collection(self, embedding_size):

        collections = [

            c.name

            for c in self.client.get_collections().collections

        ]

        if self.collection_name in collections:

            print("Deleting old collection...")

            self.client.delete_collection(

                self.collection_name

            )

        self.client.create_collection(

            collection_name=self.collection_name,

            vectors_config=VectorParams(

                size=embedding_size,

                distance=Distance.COSINE

            )

        )

        print("Fresh Collection Created.")

    # =====================================================
    # Upload
    # =====================================================

    def upload_embeddings(

        self,

        embedded_chunks,

        batch_size=QDRANT_BATCH_SIZE

    ):

        print("\nUploading Embeddings...\n")

        total = len(embedded_chunks)

        for start in tqdm(

            range(0, total, batch_size),

            desc="Uploading"

        ):

            batch = embedded_chunks[

                start:start + batch_size

            ]

            points = []

            for chunk in batch:

                payload = dict(

                    chunk["metadata"]

                )

                payload["text"] = chunk["text"]

                points.append(

                    PointStruct(

                        id=str(uuid4()),

                        vector=chunk["embedding"],

                        payload=payload

                    )

                )

            self.client.upsert(

                collection_name=self.collection_name,

                wait=True,

                points=points

            )

        print("\nUpload Completed Successfully.")

    # =====================================================
    # Search
    # =====================================================

    def search(

        self,

        query_vector,

        limit=10

    ):

        response = self.client.query_points(

            collection_name=self.collection_name,

            query=query_vector,

            limit=limit,

            with_payload=True,

            with_vectors=False

        )

        return response.points

    # =====================================================
    # Count
    # =====================================================

    def count(self):

        return self.client.count(

            collection_name=self.collection_name,

            exact=True

        ).count

    # =====================================================
    # Collection Info
    # =====================================================

    def collection_info(self):

        info = self.client.get_collection(

            self.collection_name

        )

        print("\n========== COLLECTION ==========\n")

        print("Name :", self.collection_name)

        print("Vectors :", self.count())

        print(info)

        print()