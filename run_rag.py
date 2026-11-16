import sys
import os
from pathlib import Path

# Initialize path environment
sys.path.append(str(Path(__file__).parent))

from retrieval.retriever import Retriever
from retrieval.synthesis import RAGSynthesizer

DEFAULT_QUERY = (
    "Revenue model, business model (B2B/B2C/B2G/D2C), revenue streams, pricing model, "
    "sales model, distribution model, customer acquisition model, customer interaction, "
    "OEM/direct/dealer/distributor sales, manufacturing model, outsourcing, asset-light/asset-heavy, "
    "vertical integration, operating leverage, scalability drivers, capacity utilization, "
    "recurring revenue, long-term contracts, partnerships, acquisitions, JVs, "
    "technology collaborations, ecosystem, digital platforms, recent strategic developments "
    "for ratanveer company"
)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Institutional Multimodal RAG Grounding & Synthesis Runner")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="The aspects query string to synthesize")
    parser.add_argument("--company", type=str, default=None, help="The target company ID override (e.g. 'ratnaveer')")
    args = parser.parse_args()
    
    query = args.query
    company_id = args.company

    print("=" * 70)
    print("      Institutional Multimodal RAG Grounding & Synthesis Runner      ")
    print("=" * 70)
    print(f"\nQuery:\n{query}\n")
    if company_id:
        print(f"Target Company Override: {company_id}\n")
    
    # Step 1: Initialize Retriever and Search Database
    print("Step 1: Retrieving validated chunks from database...")
    retriever = Retriever()
    if not company_id:
        company_id = retriever.detect_company_id(query)
    # Fetch initial candidates and rerank
    evidence = retriever.search(query, top_k=25, initial_fetch=250, threshold=0.40, company_id=company_id)
    
    print(f"-> Retrieved {len(evidence)} unique grounded chunks.")

    # Step 2: Initialize Synthesizer and Generate structured report
    print("\nStep 2: Synthesizing RAG Grounding Report...")
    synthesizer = RAGSynthesizer()
    report = synthesizer.generate_report(query, evidence, company_id=company_id)

    # Step 3: Write report to output file
    output_file = Path("output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n✅ Done! Structured grounding report successfully stored in: {output_file.absolute()}")
    print("\n" + "=" * 70)
    print("Summary of Report Grounding Metrics:")
    print("=" * 70)
    
    # Print the summary header from the report
    lines = report.split("\n")
    for line in lines[:15]:
        print(line)
    print("\nReview output.txt to view the full detailed aspect mapping.")
    print("=" * 70)

if __name__ == "__main__":
    main()
