import sys
from utils.statistics_uploader import StatisticsUploader
from retrieval.vector_store import VectorStore
from qdrant_client.http import models

def main():
    print("=== Verification Test: Statistics Document ===")
    
    # 1. Initialize uploader and run the upload
    uploader = StatisticsUploader()
    uploader.upload_statistics()
    
    # 2. Reuse the active VectorStore connection to prevent Qdrant local storage lock conflicts
    store = uploader.store
    
    print("\nQuerying Qdrant for document_type = 'chunk_statistics'...")
    # Scroll to retrieve the point matching our metadata filter using formal models
    results, _ = store.client.scroll(
        collection_name=store.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value="chunk_statistics")
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=True
    )
    
    # 3. Assert and verify
    if not results:
        print("\n[FAILED] Verification failed: Statistics document was not found in the collection.")
        sys.exit(1)
        
    point = results[0]
    payload = point.payload
    
    print("\n[SUCCESS] Verification successful!")
    print(f"Point ID      : {point.id}")
    print(f"Vector Length : {len(point.vector)}")
    print(f"Metadata Fields:")
    for key, val in payload.items():
        if key != "text":
            print(f"  - {key}: {val}")
            
    print(f"\nDocument Text:\n{'-'*40}\n{payload['text']}\n{'-'*40}")

if __name__ == "__main__":
    main()
