import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Initialize path environment
sys.path.append(str(Path(__file__).parent))

# Import config first to apply global monkeypatches
import config

from retrieval.retriever import Retriever
from retrieval.synthesis import RAGSynthesizer, COMPANY_SECTORS, SECTOR_MAP
from retrieval.sub_retrieval import FRIENDLY_COMPANY_NAMES
from retrieval.cache import ContextCache
from retrieval.state_manager import CheckpointManager

# Load env variables
load_dotenv()

# Define the 12 brokerage sections of the dashboard template (Detailed 209-datapoint list)
BROKERAGE_SECTIONS = [
    {
        "id": 1,
        "name": "Section 1 — Company Overview",
        "query": "Company name, history, incorporation, founders, headquarters, business description, profile details, corporate ecosystem, digital platforms, value proposition, market position for [Company Name]",
        "keywords": ["incorporation", "founders", "headquarters", "profile", "history", "founded", "ecosystem", "digital platforms"]
    },
    {
        "id": 2,
        "name": "Section 2 — Business Model",
        "query": "B2B/B2C/B2G/D2C business model, monetization model, pricing model, sales model, distribution model, customer acquisition, customer interaction, customer stickiness for [Company Name]",
        "keywords": ["business model", "monetization", "pricing model", "sales model", "customer acquisition", "customer interaction"]
    },
    {
        "id": 3,
        "name": "Section 3 — Segments & Revenue Mix",
        "query": "Product segments, revenue streams, segment revenue mix, transactional sales channels, recurring revenue, subscription ARR/MRR for [Company Name]",
        "keywords": ["segments", "revenue streams", "revenue mix", "sales channels", "recurring revenue"]
    },
    {
        "id": 4,
        "name": "Section 4 — Market Position & Moat",
        "query": "Market position, market ranking, competitive positioning, market leadership, pricing power, premium pricing, cost competitiveness, cost advantage, differentiation, proprietary technology, patents, certifications, approvals, customer approvals, switching costs, customer stickiness, barriers to entry, economies of scale, brand strength, distribution advantage, network effects, regulatory barriers, technological moat, manufacturing expertise, quality advantage for [Company Name]",
        "keywords": ["market position", "pricing power", "switching costs", "barriers to entry", "moat", "patents", "certifications", "economies of scale"]
    },
    {
        "id": 5,
        "name": "Section 5 — Key Inputs & Cost Drivers",
        "query": "Raw materials, input costs, consumables, energy/electricity utility costs, labour wages, inflation impact, commodity price volatility, foreign exchange exposure, freight costs, logistics costs for [Company Name]",
        "keywords": ["raw materials", "input costs", "consumables", "utility costs", "labour", "wages", "logistics"]
    },
    {
        "id": 6,
        "name": "Section 6 — Value Chain Positioning",
        "query": "Sourcing strategy, supplier relationships, procurement contracts, supplier concentration, import dependency, domestic sourcing, global sourcing, commodity exposure, inventory, value chain integration, vertical integration for [Company Name]",
        "keywords": ["sourcing strategy", "supplier", "procurement", "contracts", "vertical integration"]
    },
    {
        "id": 7,
        "name": "Section 7 — Manufacturing & Supply Chain",
        "query": "In-house manufacturing, battery assembly, cells sourcing, production plants, facilities, capacity, processed tonnage, SKUs, product output, automation, production volume, output, supply chain risk, logistics, distribution setup, asset-light or asset-heavy model, outsourcing for [Company Name]",
        "keywords": ["manufacturing", "production plants", "automation", "production volumes", "logistics", "outsourcing", "asset-light"]
    },
    {
        "id": 8,
        "name": "Section 8 — Customer Base & Aftermarket",
        "query": "B2B OEM customers, customer relationship, customer interaction, customer base, customer stickiness, customer concentration, aftermarket sales for [Company Name]",
        "keywords": ["customer", "stickiness", "relationship", "customer base"]
    },
    {
        "id": 9,
        "name": "Section 9 — Market Share & Dynamics",
        "query": "Market share trends, competitive positioning, market leadership dynamics, competitor dynamics for [Company Name]",
        "keywords": ["market share", "competitive positioning", "market leadership"]
    },
    {
        "id": 10,
        "name": "Section 10 — Geographic Presence",
        "query": "Geographic presence, export footprint, international trade, international offices, global distribution partners, countries served for [Company Name]",
        "keywords": ["geographic presence", "export footprint", "international", "global presence"]
    },
    {
        "id": 11,
        "name": "Section 11 — Growth Drivers",
        "query": "Scalability drivers, capacity utilization, product launches, expansion plans, capacity additions, technology investments, market opportunities for [Company Name]",
        "keywords": ["scalability drivers", "capacity utilization", "sku", "strategic developments", "expansion plans"]
    },
    {
        "id": 12,
        "name": "Section 12 — Expansion & Capital Allocation",
        "query": "Capital expenditure (capex), Phase II expansion, joint ventures, partnerships, acquisitions, technology collaborations for [Company Name]",
        "keywords": ["capex", "expansion", "joint ventures", "partnerships", "acquisitions", "collaborations"]
    }
]

# Map each Section to the exact aspect names it evaluates (Detailed 209-datapoint list)
SECTION_ASPECTS_MAP = {
    1: ["Revenue model", "Ecosystem", "Digital platforms"],
    2: ["Business model (B2B/B2C/B2G/D2C)", "Customer acquisition model", "Customer interaction"],
    3: ["Revenue streams", "Recurring revenue"],
    4: ["Pricing model"],
    5: ["Operating leverage"],
    6: ["Vertical integration"],
    7: ["Manufacturing model", "Outsourcing", "Asset-light/asset-heavy"],
    8: ["OEM/direct/dealer/distributor sales", "Sales model", "Distribution model"],
    9: ["Long-term contracts"],
    10: ["Distribution model"],
    11: ["Scalability drivers", "Capacity utilization"],
    12: ["Recent strategic developments", "Acquisitions", "JVs", "Partnerships", "Technology collaborations"]
}

def main():
    parser = argparse.ArgumentParser(description="Sequential RAG Orchestrator Runner")
    parser.add_argument("--company", type=str, default="ratnaveer", help="Target company ID (e.g. 'ratnaveer')")
    args = parser.parse_args()

    company_id = args.company.lower()
    friendly_name = FRIENDLY_COMPANY_NAMES.get(company_id, company_id.title())
    sector = COMPANY_SECTORS.get(company_id, "general")
    active_aspects = SECTOR_MAP.get(sector, SECTOR_MAP["general"])

    print("=" * 80)
    print(f"               Sequential RAG Orchestrator Runner: {friendly_name}               ")
    print("=" * 80)
    print(f"Target Company : {friendly_name} (Sector: {sector})")
    print(f"Total Sections : {len(BROKERAGE_SECTIONS)}")
    print("-" * 80)

    # Initialize RAG components
    retriever = Retriever()
    synthesizer = RAGSynthesizer()

    # Load state checkpoint if exists (Problem 2 / Milestone 2)
    checkpoint = CheckpointManager.load_checkpoint(company_id)
    completed_sections = checkpoint.get("completed_sections", [])
    section_outputs = {int(k): v for k, v in checkpoint.get("section_outputs", {}).items()}

    if completed_sections:
        print(f"[Checkpoint] Loaded existing progress. Resuming from last checkpoint (Completed: {completed_sections}).")

    report_sections = []
    
    for section in BROKERAGE_SECTIONS:
        sec_name = section["name"]
        sec_query = section["query"].replace("[Company Name]", friendly_name)
        sec_id = section["id"]

        print(f"\n[{sec_name}] starting...")

        # Restore from checkpoint if available
        if sec_id in completed_sections:
            print(f"  [Checkpoint] Section already completed. Restored from state checkpoint.")
            report_sections.append(section_outputs[sec_id])
            continue

        # 1. Sector-based Pre-filtering Check (based on mapped aspects)
        is_inapplicable = False
        mapped_aspects = SECTION_ASPECTS_MAP.get(sec_id, [])
        if mapped_aspects:
            # It is inapplicable only if ALL its mapped aspects are default_not_applicable!
            all_inapplicable = True
            for asp_name in mapped_aspects:
                if asp_name in active_aspects:
                    if not active_aspects[asp_name]["default_not_applicable"]:
                        all_inapplicable = False
                        break
            if all_inapplicable:
                is_inapplicable = True

        if is_inapplicable:
            print(f"  [Pre-Filtering] Pruned section - Inapplicable for sector '{sector}'. Skipping DB retrieval.")
            na_text = f"## {sec_name}\n\nNot Applicable / Not mentioned in filings (The {sector} business model typically does not utilize these aspects).\n"
            report_sections.append(na_text)
            
            # Save to checkpoint state
            completed_sections.append(sec_id)
            section_outputs[sec_id] = na_text
            CheckpointManager.save_checkpoint(company_id, completed_sections, section_outputs)
            continue

        # 2. Database Retrieval & Synthesis with Auto-Healing Retry Loop
        max_retries = 3
        evidence = []
        section_report = ""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Retrieving validated chunks (Attempt {attempt}/{max_retries})...")
                evidence = retriever.search(sec_query, top_k=8, initial_fetch=120, threshold=0.40, company_id=company_id)
                print(f"  Retrieved {len(evidence)} chunks.")

                # Store Context in Cache (Pass-by-Reference)
                ref_keys = ContextCache.store(evidence)
                print(f"  [Context Cache] Stored {len(evidence)} chunks (Keys: {ref_keys})")

                # LLM Synthesis by Reference
                print("  Synthesizing section text...")
                section_report = synthesizer.generate_report_by_ref(sec_query, ref_keys, company_id=company_id)
                break
            except Exception as e:
                print(f"  Warning: Attempt {attempt} failed ({e}).")
                if attempt < max_retries:
                    sleep_time = 5 * attempt
                    print(f"  Auto-Healing: Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("  Auto-Healing Error: All retries failed. Degraded to offline rules fallback.")
                    section_report = synthesizer._generate_with_rules(sec_query, evidence, company_id=company_id)
        
        # Format the section nicely in Markdown
        formatted_section = f"## {sec_name}\n\n{section_report.strip()}\n"
        report_sections.append(formatted_section)

        # Save to checkpoint state
        completed_sections.append(sec_id)
        section_outputs[sec_id] = formatted_section
        CheckpointManager.save_checkpoint(company_id, completed_sections, section_outputs)
        print(f"  [Checkpoint] State persisted for Section {sec_id}.")

        # 4. Respect Rate Limits (2.5 seconds pause between queries)
        print("  Cooldown sleep (2.5s)...")
        time.sleep(2.5)

    # Assemble and write the final compiled report
    print("\n" + "=" * 80)
    print("Writing compiled report to output.txt...")
    print("=" * 80)

    final_report = f"# Institutional Equity Research Report: {friendly_name}\n\n"
    final_report += "\n\n".join(report_sections)

    output_path = Path("output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    # Clear checkpoint upon successful completion of full report
    CheckpointManager.clear_checkpoint(company_id)

    print(f"\n✅ Success! Compiled 12-section research report written to: {output_path.absolute()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
