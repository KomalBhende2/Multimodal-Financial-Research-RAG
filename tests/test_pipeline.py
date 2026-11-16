from pathlib import Path
import ingestion.models.document_schema as schema

print(schema.__file__)
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

output = pipeline.process_pdf(
    Path("data/documents/Q1FY26 PPT.pdf")
)

print()

print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(output)