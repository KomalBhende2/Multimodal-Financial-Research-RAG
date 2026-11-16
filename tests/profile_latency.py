import sys
import os
import time
import psycopg2
from pathlib import Path
import gc
import torch

# Add root folder to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import config
from retrieval.retriever import Retriever
from sentence_transformers import SentenceTransformer, CrossEncoder
from evidence.validator import EvidenceValidator
from retrieval.sub_retrieval import deduplicate_evidence
from models.evidence import Evidence

def profile_retrieval():
    query = (
        "Raw materials, components, consumables, energy, utilities, labour, logistics costs, "
        "freight costs, sourcing strategy, supplier base, supplier concentration, import dependency, "
        "domestic sourcing, global sourcing, commodity exposure, commodity price volatility, "
        "foreign exchange exposure, supply chain risk, inventory, procurement strategy, "
        "cost inflation, margin drivers, gross margin impact, mitigation strategies, "
        "long-term supply agreements for ratanveer company"
    )
    company_id = "ratnaveer"
    
    timings = {}
    
    t_start = time.time()
    
    # 1. Embedding load
    t0 = time.time()
    emb_model = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
    timings["Embedding load"] = time.time() - t0
    
    # 2. Embedding generation
    t0 = time.time()
    query_embedding = emb_model.encode(query).tolist()
    timings["Embedding generation"] = time.time() - t0
    
    del emb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    retriever = Retriever()
    
    # Run execute_hybrid_query which internally runs Vector search, Lexical search, and Merging.
    # We will measure the database times and merge times from the retriever's internally tracked diagnostics!
    t0 = time.time()
    rows = retriever.execute_hybrid_query(query, company_id, query_embedding, limit=120)
    total_db_time = time.time() - t0
    
    # Extract database-level breakdowns
    timings["Vector search"] = retriever.last_query_diagnostics.get("vector_latency", 0.0)
    timings["Lexical search"] = retriever.last_query_diagnostics.get("lexical_latency", 0.0)
    timings["Merge"] = total_db_time - timings["Vector search"] - timings["Lexical search"]
    
    # Keep only the top RERANK_LIMIT candidates (pre-filtering)
    from config import RERANK_LIMIT
    rows_to_process = rows[:RERANK_LIMIT]
    
    # Map to Evidence objects
    evidence_list = []
    for chunk_id, payload, similarity, emb_vector in rows_to_process:
        ev = Evidence(
            company=payload.get("company_name", "Ratnaveer"),
            document=payload.get("document_type", "Annual Report"),
            source_type=payload.get("chunk_type", "text"),
            report_year=2026,
            page=int(payload.get("page_number", 0)),
            chunk_id=chunk_id,
            text=payload.get("text", ""),
            embedding=emb_vector,
            retrieval_score=similarity
        )
        ev.topic = payload.get("topic", "general")
        ev.document_scope = payload.get("document_scope", "current")
        evidence_list.append(ev)

    # 3. CrossEncoder load
    t0 = time.time()
    reranker = CrossEncoder(config.RERANKER_MODEL)
    timings["CrossEncoder load"] = time.time() - t0
    
    # 4. CrossEncoder inference
    t0 = time.time()
    pairs = [[query, ev.text] for ev in evidence_list]
    rerank_scores = reranker.predict(pairs, activation_fn=lambda x: x)
    for ev, score in zip(evidence_list, rerank_scores):
        ev.retrieval_score = float(score)
    timings["CrossEncoder inference"] = time.time() - t0
    
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 5. Validation
    t0 = time.time()
    validator = EvidenceValidator()
    valid_evidence = validator.filter_valid(query, evidence_list)
    timings["Validation"] = time.time() - t0
    
    # 6. Deduplication (lazy loading SentenceTransformer - skips loading because all candidates have embeddings populated!)
    t0 = time.time()
    needs_model = any(not getattr(ev, "embedding", None) for ev in valid_evidence)
    emb_model2 = None
    if needs_model:
        emb_model2 = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
    deduplicated = deduplicate_evidence(valid_evidence, emb_model2)
    timings["Deduplication"] = time.time() - t0
    
    if emb_model2:
        del emb_model2
        gc.collect()
        
    timings["Total"] = time.time() - t_start
    
    # ======================================================================
    # PRINT RESULTS
    # ======================================================================
    print("\n" + "="*50)
    print("      RAG RETRIEVAL LATENCY BREAKDOWN REPORT      ")
    print("="*50)
    print(f"Query: '{query[:60]}...'")
    print(f"Company ID: '{company_id}' | Candidates Retrieved: {len(rows)}")
    print(f"Candidates Reranked: {len(evidence_list)} | Candidates Kept: {len(deduplicated)}")
    print("-"*50)
    
    for phase, duration in timings.items():
        print(f"{phase:<28} : {duration * 1000:>10.2f} ms ({duration:>6.3f} s)")
        
    print("="*50)

if __name__ == "__main__":
    profile_retrieval()
