import sys
from pathlib import Path

# Add root folder to sys.path to allow imports from root modules
sys.path.append(str(Path(__file__).parent.parent))

# Import config first to apply global SSL/Unicode patches
import config
from retrieval.retriever import Retriever
from retrieval.synthesis import RAGSynthesizer

def main():
    print("=== Qdrant/Postgres Semantic Retrieval CLI ===")
    
    # 1. Initialize retriever and synthesizer
    retriever = Retriever()
    synthesizer = RAGSynthesizer()

    # 2. Check for custom query passed as command-line arguments
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        # Default RAG queries
        queries = [
            "What competitive advantages position the company for long-term growth? Include technology, research and development, manufacturing capabilities, distribution network, brand strength, and market share.",
            "What are the key factors expected to drive the company's future growth over the next 3–5 years? Include expansion plans, product launches, capacity additions, technology investments, and market opportunities.",
            "Identify the major revenue and profitability growth drivers for the company. Explain how new products, geographic expansion, pricing strategy, operating leverage, and margin improvements are expected to impact future financial performance."
        ]

    # 3. Execute search queries and print the results
    for idx, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {idx}: {query}")
        print("=" * 80)
        
        # Search Postgres, rerank, and threshold (fetch 80, rerank to top 20, threshold 0.45)
        company_id = retriever.detect_company_id(query)
        results = retriever.search(query, top_k=20, initial_fetch=80, threshold=0.45, company_id=company_id)
        
        # 4. Generate the Grounding Metrics Scorecard
        print("\n" + "-" * 80)
        print("GROUNDING METRICS SCORECARD")
        print("-" * 80)
        scorecard = synthesizer._generate_with_rules(query, results, company_id=company_id)
        print(scorecard)
        
        print("\n" + "-" * 80)
        print(f"Found {len(results)} relevant evidence chunks after reranking:")
        print("-" * 80)
        for res_idx, ev in enumerate(results, 1):
            print(f"\n  [Result {res_idx}] Similarity Score: {ev.retrieval_score:.4f} | Page: {ev.page} | Chunk Type: {ev.source_type}")
            print(f"  Heading: {ev.company} - {ev.document}")
            print(f"  Content snippet:\n  {'-'*60}\n  {ev.text.strip()}\n  {'-'*60}")
        print()

if __name__ == "__main__":
    main()
