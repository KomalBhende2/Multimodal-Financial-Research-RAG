import json
import psycopg2
import numpy as np
import copy
import gc
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import EMBEDDING_MODEL, RERANKER_MODEL
from typing import List, Optional
from models.evidence import Evidence

# Configurable number of chunks retrieved per section
SECTION_TOP_K = 8

# Mapping of company IDs to friendly search prefixes
FRIENDLY_COMPANY_NAMES = {
    "azad_engineering": "Azad Engineering",
    "bajaj_finance": "Bajaj Finance",
    "pine_labs": "Pine Labs",
    "hscl": "HSCL Himadri",
    "bank_of_maharashtra": "Bank of Maharashtra",
    "ratnaveer": "Ratnaveer",
    "ather": "Ather Energy"
}

# 5 logical theme-based semantic sections
THEME_SECTIONS = [
    {
        "name": "Corporate Identity",
        "keywords": [
            "company name", "incorporation", "founders", "headquarters", "business description", 
            "overview", "about us", "profile", "about the company", "snapshot", "at a glance", 
            "history", "founded", "value proposition", "market position", "vertical integration",
            "ecosystem", "digital platforms"
        ],
        "template": "company overview, history, incorporation, founders, headquarters, business description, value proposition, market position, vertical integration"
    },
    {
        "name": "Business & Products",
        "keywords": [
            "product", "service", "portfolio", "industry", "sector", "business type", 
            "customer", "plant", "office", "launch", "patent", "manufacturing", "facilities", "capabilities",
            "pricing", "pricing model", "distribution", "distribution model", "sales model", "outsourcing",
            "asset-light", "asset-heavy", "customer acquisition", "customer interaction", "direct sales",
            "dealer", "distributor", "sourcing strategy", "labour", "supplier", "supplier base", "procurement", "components"
        ],
        "template": "business overview, products and services, manufacturing capabilities, production facilities, customer segments, patents, pricing model, sales distribution, outsourcing, asset-light or asset-heavy model, sourcing strategy, labour requirements"
    },
    {
        "name": "Financial Performance",
        "keywords": [
            "profitability", "revenue", "margin", "scale", "milestones", "ebitda", "pat", 
            "financial", "income", "cost", "operating", "leverage", "operating leverage",
            "recurring revenue", "revenue model", "revenue streams", "profitability trend"
        ],
        "template": "financial performance, revenue trend, profitability, EBITDA margin, PAT, operating scale, financial milestones, operating leverage, revenue model"
    },
    {
        "name": "Corporate Events",
        "keywords": [
            "evolution", "milestones", "ipo", "acquisitions", "mergers", "jv", "partnership", "strategic", "events",
            "joint venture", "collaboration", "contracts", "long-term contracts", "acquisitions", "partnerships", "jvs",
            "supply agreements", "supply contracts"
        ],
        "template": "business evolution, major milestones, timeline, IPO listing, acquisitions, mergers, JVs, partnerships, joint ventures, collaborations, long-term contracts, supply agreements"
    },
    {
        "name": "Global Presence & Compliance",
        "keywords": [
            "export", "international", "global", "presence", "footprint", "certifications", 
            "approvals", "accreditation", "regulatory"
        ],
        "template": "global presence, export footprint, certifications, regulatory approvals, international offices, accreditations"
    },
    {
        "name": "Operating Volumes & Metrics",
        "keywords": [
            "capacity", "processed", "mt", "sku", "skus", "production volume", "output",
            "raw materials", "consumables", "inventory", "utilities", "energy"
        ],
        "template": "annual production volumes, metric tonnes processed, capacity utilization, SKUs, product output, raw materials, inventory, energy requirements"
    },
    {
        "name": "Subsidiaries & Governance",
        "keywords": [
            "subsidiary", "subsidiaries", "governance", "whistle blower", "sharjah", "media city", "foreign subsidiary"
        ],
        "template": "corporate structure, whistle blower policy, foreign subsidiaries, Sharjah office, board committees, Vigil Mechanism"
    }
]

def detect_broad_query(query: str) -> bool:
    """
    Returns True if the query requests information spanning 3 or more distinct
    theme-based semantic sections, indicating it is a broad, multi-aspect query.
    """
    q = query.lower()
    matched_themes = 0
    for theme in THEME_SECTIONS:
        if any(kw in q for kw in theme["keywords"]):
            matched_themes += 1
    return matched_themes >= 3

def generate_semantic_sub_queries(query: str, company_name: str, company_id: str = None) -> List[str]:
    """
    Analyzes the query and generates clean, distinct semantic sub-queries for each requested theme,
    incorporating specific matched query terms to steer vector search.
    Apply dynamic sector-based pre-filtering to prune inapplicable search tasks (Problem 1 / Reason 3).
    """
    q = query.lower()
    sub_queries = []
    
    # Pre-filtering logic based on company sector config
    from retrieval.synthesis import COMPANY_SECTORS, SECTOR_MAP
    sector = COMPANY_SECTORS.get(company_id, "general")
    active_aspects = SECTOR_MAP.get(sector, SECTOR_MAP["general"])
    
    for theme in THEME_SECTIONS:
        matched_keywords = [kw for kw in theme["keywords"] if kw in q]
        if matched_keywords:
            # Filter out any keywords that belong to aspects marked as default_not_applicable for this sector
            applicable_keywords = []
            for kw in matched_keywords:
                kw_inapplicable = False
                for aspect_name, config in active_aspects.items():
                    if kw in config["keywords"] and config["default_not_applicable"]:
                        kw_inapplicable = True
                        break
                if not kw_inapplicable:
                    applicable_keywords.append(kw)
            
            if not applicable_keywords:
                print(f"  [Pre-Filtering] Pruned theme section '{theme['name']}' - All matched keywords are inapplicable to sector '{sector}'")
                continue
                
            specific_terms = ", ".join(applicable_keywords)
            sub_query = f"{company_name} {theme['template']} (Focus: {specific_terms})" if company_name else f"{theme['template']} (Focus: {specific_terms})"
            sub_queries.append((theme["name"], sub_query))
            
    return sub_queries

def retrieve_section_candidates(retriever, section_query: str, company_id: str, embedding_model, fetch_limit: int = 120) -> List[Evidence]:
    """
    Executes a hybrid search query strictly filtered by company_id for a single section query.
    """
    query_embedding = embedding_model.encode(section_query).tolist()
    
    # Execute modular hybrid query from retriever
    rows = retriever.execute_hybrid_query(section_query, company_id, query_embedding, limit=fetch_limit)
    
    candidates = []
    for chunk_id, payload, similarity, emb_vector in rows:
        
        company = payload.get("company_name") or payload.get("company_id") or "Ather Energy"
        document = payload.get("document_type") or payload.get("doc_type") or "Investor Presentation"
        source_type = payload.get("chunk_type") or payload.get("source_type") or "unknown"
        page = payload.get("page_number") or payload.get("page") or 0
        text_body = payload.get("text") or payload.get("page_content") or ""
        
        heading = payload.get("heading") or payload.get("section_heading") or payload.get("section") or "general"
        topic = payload.get("topic") or payload.get("section_heading") or "general"
        document_scope = payload.get("document_scope") or "current"
        
        friendly_company = company.replace("_", " ").title() if isinstance(company, str) else str(company)
        friendly_document = document.replace("_", " ").title() if isinstance(document, str) else str(document)
        
        ev = Evidence(
            company=friendly_company,
            document=friendly_document,
            source_type=source_type,
            report_year=2026,
            page=int(page),
            chunk_id=chunk_id,
            text=text_body,
            embedding=emb_vector,
            retrieval_score=similarity
        )
        ev.topic = topic
        ev.document_scope = document_scope
        ev.db_company_id = payload.get("company_id") or company_id
        ev.doc_id = payload.get("doc_id") or payload.get("document_id")
        ev.chunk_index = payload.get("chunk_index")
        ev.section = payload.get("section") or payload.get("heading") or payload.get("section_heading") or ""
        candidates.append(ev)
        
    return candidates

def deduplicate_evidence(evidence_list: List[Evidence], embedding_model=None) -> List[Evidence]:
    """
    Deduplicates merged chunks based on page mapping, Jaccard similarity, and Cosine similarity.
    Pre-sorts by text length to ensure the richest content chunk is preserved.
    """
    # Keep the richest content first by sorting by length descending
    evidence_list.sort(key=lambda x: len(x.text), reverse=True)
    
    unique_evidence = []
    seen_pages = set()
    selected_embeddings = []
    
    print("\n--- Deduplication Trace ---")
    for ev in evidence_list:
        page_key = (ev.document, ev.page)
        if page_key in seen_pages:
            print(f"  [Deduplication] Rejected Page {ev.page} ({ev.document}) - Duplicate Page Key.")
            continue
            
        # Jaccard Check (Lexical Redundancy)
        is_redundant = False
        words_curr = set(ev.text.lower().split())
        for sel_ev in unique_evidence:
            words_sel = set(sel_ev.text.lower().split())
            intersection = words_curr.intersection(words_sel)
            union = words_curr.union(words_sel)
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0.55:
                is_redundant = True
                print(f"  [Deduplication] Rejected Page {ev.page} ({ev.document}) - Jaccard redundancy ({jaccard:.2f}) with Page {sel_ev.page} ({sel_ev.document}).")
                break
        if is_redundant:
            continue
            
        # Cosine Check (Semantic Redundancy)
        if getattr(ev, "embedding", None) and len(ev.embedding) > 0:
            ev_embedding = np.array(ev.embedding)
        elif embedding_model is not None:
            ev_embedding = embedding_model.encode(ev.text)
        else:
            ev_embedding = None
            
        is_semantically_redundant = False
        if ev_embedding is not None:
            for sel_ev, sel_emb in zip(unique_evidence, selected_embeddings):
                if sel_emb is None:
                    continue
                sim = np.dot(ev_embedding, sel_emb) / (np.linalg.norm(ev_embedding) * np.linalg.norm(sel_emb))
                if sim > 0.88:
                    is_semantically_redundant = True
                    print(f"  [Deduplication] Rejected Page {ev.page} ({ev.document}) - Cosine redundancy ({sim:.2f}) with Page {sel_ev.page} ({sel_ev.document}).")
                    break
        if is_semantically_redundant:
            continue
            
        print(f"  [Deduplication] KEPT: Page {ev.page} ({ev.document}) - unique content.")
        unique_evidence.append(ev)
        seen_pages.add(page_key)
        selected_embeddings.append(ev_embedding)
        
    # Re-sort by similarity score descending so that standard CrossEncoder reranking is unaffected
    unique_evidence.sort(key=lambda x: x.retrieval_score, reverse=True)
    return unique_evidence

def decompose_and_retrieve(retriever, query: str, company_id: str, threshold: float = 0.50, section_top_k: int = SECTION_TOP_K) -> Optional[List[Evidence]]:
    """
    Decomposes a broad query into semantic theme sections, retrieves candidates strictly
    filtered by company_id for each section, and merges/deduplicates the results.
    """
    if not company_id:
        company_id = retriever.detect_company_id(query)

    if not detect_broad_query(query):
        return None
        
    friendly_company = FRIENDLY_COMPANY_NAMES.get(company_id, "")
    sections = generate_semantic_sub_queries(query, friendly_company, company_id=company_id)
    
    if not sections:
        return None
        
    print(f"\n[Query Decomposition] Detected broad query. Generated {len(sections)} theme-based sections:")
    for name, sq in sections:
        print(f"  Theme: '{name}' -> Query: '{sq}'")
        
    from evidence.validator import EvidenceValidator
    from reports.coverage_reporter import CoverageReporter
    validator = EvidenceValidator()
    
    # ----------------------------------------------------
    # PHASE 1: Retrieve candidates for all sections using embedding model
    # ----------------------------------------------------
    print("\n--- Phase 1: Embedding & Retrieval ---")
    print("Loading Embedding Model for section queries...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    sections_candidates = {}
    try:
        for idx, (name, sq) in enumerate(sections, 1):
            print(f"\nRetrieving candidates for Theme Section {idx} ({name}): '{sq}'")
            candidates = retrieve_section_candidates(retriever, sq, company_id, embedding_model, fetch_limit=120)
            print(f"Database returned {len(candidates)} candidates.")
            sections_candidates[(name, sq)] = candidates
    finally:
        del embedding_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------
    # PHASE 2: Rerank and filter candidates using reranker model
    # ----------------------------------------------------
    print("\n--- Phase 2: Reranking & Scoring ---")
    print("Loading Reranker Model for sub-query reranking...")
    import time
    retriever.last_query_diagnostics["reranking_latency"] = 0.0
    reranker_model = CrossEncoder(RERANKER_MODEL)
    merged_candidates = []
    try:
        for idx, ((name, sq), candidates) in enumerate(sections_candidates.items(), 1):
            if not candidates:
                continue
                
            # Log company_id distribution before merging
            dist = {}
            for c in candidates:
                cid = getattr(c, "db_company_id", "unknown")
                dist[cid] = dist.get(cid, 0) + 1
            print(f"  [Company ID Distribution] Before merging for '{name}': {dist}")
            
            # Pre-filter each section's candidates before CrossEncoder reranking
            from config import RERANK_LIMIT_SECTION
            candidates_to_rerank = candidates[:RERANK_LIMIT_SECTION]

            # Local Cross-Encoder reranking for this specific section query
            pairs = [[sq, ev.text] for ev in candidates_to_rerank]
            t_pr0 = time.time()
            scores = reranker_model.predict(pairs, activation_fn=lambda x: x)
            t_pr1 = time.time()
            retriever.last_query_diagnostics["reranking_latency"] += (t_pr1 - t_pr0)
            
            # Score computation & Penalization
            for ev, score in zip(candidates_to_rerank, scores):
                boosted_score = float(score)
                
                # Apply general penalty for detailed accounting policies & notes
                text_lower = ev.text.lower()
                if any(k in text_lower for k in ["notes forming part of", "accounting policies", "basis of preparation", "accounting standards", "statement of compliance"]):
                    if not any(k in text_lower for k in ["registered office", "company information"]):
                        if not any(k in query.lower() for k in ["accounting", "notes", "policies", "basis of preparation", "compliance"]):
                            boosted_score -= 4.5
                        
                # Penalize known cover/disclaimer/signature noise pages
                if ev.page in {1, 2, 20} or "disclaimer" in text_lower or "puja aggarwal" in text_lower:
                    boosted_score -= 3.0
                    
                # Direct target boost for corporate info and board registry pages
                if any(k in text_lower for k in ["board of directors", "key management personnel", "corporate information"]):
                    if any(q_word in query.lower() for q_word in ["founder", "headquarters", "director", "promoter", "management", "governance", "chairman", "company name"]):
                        boosted_score += 5.0
                        
                if any(k in text_lower for k in ["registered office"]):
                    if any(q_word in query.lower() for q_word in ["headquarters", "office", "registered office", "location", "address", "company name"]):
                        boosted_score += 5.0
                        
                if any(k in text_lower for k in ["founding promoter", "promoter details"]):
                    if any(q_word in query.lower() for q_word in ["founder", "promoter", "management", "company name"]):
                        boosted_score += 5.0

                # Direct target boost for corporate transactions (acquisitions, mergers, JVs, partnerships)
                if any(k in text_lower for k in ["acquired", "acquisition", "merger", "joint venture", "jv", "partnership"]):
                    if any(q_word in query.lower() for q_word in ["acquisition", "merger", "jv", "joint venture", "partnership", "acquire"]):
                        boosted_score += 4.0

                # Direct target boost for global presence and subsidiaries
                if any(k in text_lower for k in ["subsidiary", "geographical footprint", "export markets", "international footprint"]):
                    if any(q_word in query.lower() for q_word in ["subsidiary", "export", "global presence", "footprint", "international office"]):
                        boosted_score += 4.0

                # Timeline & Journey Boost
                if any(k in text_lower for k in ["highlights of our journey", "journey", "milestones", "milestone", "chronology", "timeline"]):
                    if any(q_word in query.lower() for q_word in ["timeline", "milestone", "journey", "evolution", "history", "ipo"]):
                        boosted_score += 5.0
                    
                try:
                    import math
                    clamped_score = max(-50.0, min(50.0, boosted_score))
                    sigmoid_score = 1.0 / (1.0 + math.exp(-(clamped_score + 7.2) / 2.0))
                except Exception:
                    sigmoid_score = float(boosted_score)
                    
                ev.retrieval_score = sigmoid_score
                
            # Filter with similarity threshold
            filtered_candidates = [ev for ev in candidates_to_rerank if ev.retrieval_score >= threshold]
            print(f"  [Threshold Filter] {len(filtered_candidates)} candidates passed threshold (>= {threshold}).")
            
            # Validate chunks
            valid_candidates = validator.filter_valid(sq, filtered_candidates)
            print(f"  [Validator Filter] {len(valid_candidates)} candidates passed validation.")
            
            # Sort and take top section_top_k chunks
            valid_candidates.sort(key=lambda x: x.retrieval_score, reverse=True)
            section_top_chunks = valid_candidates[:section_top_k]
            
            merged_candidates.extend(section_top_chunks)
            print(f"  Selected top {len(section_top_chunks)} chunks for Theme Section {idx} ({name}):")
            for sc in section_top_chunks:
                print(f"    - Page {sc.page} ({sc.document}) | Score: {sc.retrieval_score:.4f}")
    finally:
        del reranker_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[Merge & Deduplication] Merged pool size: {len(merged_candidates)} chunks.")
    
    # ----------------------------------------------------
    # PHASE 3: Deduplicate merged candidates using embedding model
    # ----------------------------------------------------
    print("\n--- Phase 3: Deduplication ---")
    needs_model = any(not getattr(ev, "embedding", None) for ev in merged_candidates)
    embedding_model = None
    if needs_model:
        print("Loading Embedding Model for deduplication redundancy checks...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    try:
        deduplicated = deduplicate_evidence(merged_candidates, embedding_model)
    finally:
        if embedding_model:
            del embedding_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    initial_cnt = len(merged_candidates)
    final_cnt = len(deduplicated)
    dup_rate = (initial_cnt - final_cnt) / initial_cnt if initial_cnt > 0 else 0.0
    retriever.last_query_diagnostics["duplicate_rate"] = dup_rate

    print(f"[Merge & Deduplication] Deduplicated pool size: {len(deduplicated)} chunks.")
    
    # Adjacent Chunk Expansion (Post-rerank, post-validation, post-coverage, post-deduplication)
    from config import ENABLE_ADJACENT_EXPANSION
    if ENABLE_ADJACENT_EXPANSION:
        deduplicated = retriever.expand_adjacent_chunks(deduplicated)
    
    # Generate coverage report on the final deduplicated merged list
    requested_themes = retriever.identify_requested_themes(query)
    if requested_themes:
        reporter = CoverageReporter()
        report_content = reporter.generate_report(query, requested_themes, deduplicated, retriever)
        reporter.save_report_to_file(query, report_content)
        
    return deduplicated
