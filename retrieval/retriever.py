"""
Retriever
----------
Retrieves the most relevant evidence from PostgreSQL/PGVector and reranks them.
"""

import math
import json
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc
import torch
from config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    VECTOR_WEIGHT, LEXICAL_WEIGHT,
    VECTOR_SEARCH_LIMIT, KEYWORD_SEARCH_LIMIT,
    BOOST_COMPANY, BOOST_DOCUMENT, BOOST_TOPIC, BOOST_QUARTER, BOOST_SECTION,
    THRESHOLD_SHORT_QUERY, THRESHOLD_MEDIUM_QUERY, THRESHOLD_LONG_QUERY,
    SHORT_QUERY_MAX_WORDS, MEDIUM_QUERY_MAX_WORDS
)
from models.evidence import Evidence
import re
import os

FINANCIAL_SYNONYMS = {
    "outsourcing": ["job work", "sub-contracting", "subcontracting", "third-party processing"],
    "pricing power": ["premium pricing", "price hikes", "pass-through", "price increase"],
    "recurring revenue": ["subscription", "arr", "mrr", "licensing fees"],
    "contracts": ["agreements", "supply agreements", "purchase agreements", "mou"],
    "acquisitions": ["takeover", "merger", "amalgamation", "sub-lease"],
    "jvs": ["joint venture", "joint ventures", "associate", "associates"],
    "regulatory barriers": ["approvals", "certifications", "licensing", "compliance", "ecms"],
    "geography": ["exports", "international", "global", "outside india", "shipping"],
    "capital allocation": ["capex", "investments", "capital expenditure", "fundraise"]
}

def expand_query_keywords(query: str) -> list:
    query_lower = query.lower()
    expanded_tokens = []
    
    # 1. Dictionary-first lookup
    for phrase, synonyms in FINANCIAL_SYNONYMS.items():
        if phrase in query_lower:
            for syn in synonyms:
                # Split multi-word synonyms into clean words
                for word in re.split(r'[\s\-]+', syn.lower()):
                    w = word.strip()
                    if w and len(w) > 2:
                        expanded_tokens.append(w)
                        
    # 2. Optional LLM expansion if enabled and dictionary didn't yield enough tokens
    if os.environ.get("ENABLE_LLM_QUERY_EXPANSION", "false").lower() == "true":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
                model = genai.GenerativeModel(model_name)
                
                prompt = f"""
You are a financial search query assistant. Expand the following institutional query with 3-4 highly specific financial accounting synonyms or related keywords (e.g. "outsourcing" -> "job work", "subcontracting").
Query: {query}
Return only a comma-separated list of synonym words. Do not output anything else.
"""
                response = model.generate_content(prompt)
                for term in response.text.split(","):
                    for word in re.split(r'[\s\-]+', term.lower()):
                        w = word.strip()
                        if w and len(w) > 2:
                            expanded_tokens.append(w)
            except Exception as e:
                print(f"Warning: Optional LLM Query Expansion failed ({e}).")
                
    return list(set(expanded_tokens))


class Retriever:

    def __init__(self):
        self.db_url = "postgresql://postgres:postgre123@127.0.0.1:5432/findoctest"
        self.last_query_diagnostics = {}

    def execute_hybrid_query(self, query: str, company_id: str, query_embedding: list, limit: int = None) -> list:
        """
        Executes a production-grade Hybrid Retrieval querying PostgreSQL for both 
        Dense PGVector candidates and Lexical Full-Text Search candidates, merging 
        and boosting them using metadata matches.
        """
        import re
        import time

        self.last_query_diagnostics["vector_latency"] = 0.0
        self.last_query_diagnostics["lexical_latency"] = 0.0
        self.last_query_diagnostics["hybrid_contribution"] = {}

        v_limit = limit if limit is not None else VECTOR_SEARCH_LIMIT
        k_limit = limit if limit is not None else KEYWORD_SEARCH_LIMIT

        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        # 1. Execute PGVector Dense Search
        t_v0 = time.time()
        if company_id:
            vector_sql = """
                SELECT id, payload, embedding::text, 1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE company_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cursor.execute(vector_sql, (query_embedding, company_id, query_embedding, v_limit))
        else:
            vector_sql = """
                SELECT id, payload, embedding::text, 1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cursor.execute(vector_sql, (query_embedding, query_embedding, v_limit))
        
        vector_rows = cursor.fetchall()
        t_v1 = time.time()
        self.last_query_diagnostics["vector_latency"] = t_v1 - t_v0

        # 2. Execute PostgreSQL Full Text Search (FTS)
        t_l0 = time.time()
        lexical_rows = []
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if clean_query:
            words = [w.strip() for w in clean_query.lower().split() if len(w.strip()) > 3]
            stop_words = {"what", "where", "when", "how", "who", "which", "why", "the", "and", "for", "include", "identify", "explain", "expected", "company", "position", "positioning", "latest"}
            keywords = [w for w in words if w not in stop_words]
            
            # Apply dynamic query expansions (Problem 4 / Terminology Gaps)
            expanded = expand_query_keywords(query)
            if expanded:
                # Add expanded synonyms to keyword list
                keywords = list(set(keywords + expanded))
            
            if keywords:
                tsquery_str = " | ".join(keywords)
                if company_id:
                    fts_sql = """
                        SELECT id, payload, embedding::text, ts_rank_cd(
                            to_tsvector('english', 
                                coalesce(payload->>'text', '') || ' ' || 
                                coalesce(payload->>'heading', '') || ' ' || 
                                coalesce(payload->>'topic', '') || ' ' || 
                                coalesce(payload->>'section', '')
                            ), 
                            to_tsquery('english', %s)
                        ) AS rank
                        FROM chunks
                        WHERE company_id = %s
                        AND to_tsvector('english', 
                            coalesce(payload->>'text', '') || ' ' || 
                            coalesce(payload->>'heading', '') || ' ' || 
                            coalesce(payload->>'topic', '') || ' ' || 
                            coalesce(payload->>'section', '')
                        ) @@ to_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """
                    cursor.execute(fts_sql, (tsquery_str, company_id, tsquery_str, k_limit))
                else:
                    fts_sql = """
                        SELECT id, payload, embedding::text, ts_rank_cd(
                            to_tsvector('english', 
                                coalesce(payload->>'text', '') || ' ' || 
                                coalesce(payload->>'heading', '') || ' ' || 
                                coalesce(payload->>'topic', '') || ' ' || 
                                coalesce(payload->>'section', '')
                            ), 
                            to_tsquery('english', %s)
                        ) AS rank
                        FROM chunks
                        WHERE to_tsvector('english', 
                            coalesce(payload->>'text', '') || ' ' || 
                            coalesce(payload->>'heading', '') || ' ' || 
                            coalesce(payload->>'topic', '') || ' ' || 
                            coalesce(payload->>'section', '')
                        ) @@ to_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """
                    cursor.execute(fts_sql, (tsquery_str, tsquery_str, k_limit))
                
                lexical_rows = cursor.fetchall()
        t_l1 = time.time()
        self.last_query_diagnostics["lexical_latency"] = t_l1 - t_l0

        cursor.close()
        conn.close()

        # 3. Merge candidates and remove duplicate chunk IDs
        merged_candidates = {}

        # Add vector results
        for row in vector_rows:
            chunk_id, payload_data, embedding_text, similarity = row
            payload = payload_data if isinstance(payload_data, dict) else json.loads(payload_data)
            
            # Convert embedding text to list of floats
            emb_vector = []
            if embedding_text and isinstance(embedding_text, str):
                emb_vector = [float(x) for x in embedding_text.strip('[]').split(',') if x.strip()]
                
            merged_candidates[chunk_id] = {
                "payload": payload,
                "embedding": emb_vector,
                "vector_score": float(similarity) if similarity is not None else 0.0,
                "lexical_score": 0.0
            }

        # Add/update lexical results
        for row in lexical_rows:
            chunk_id, payload_data, embedding_text, rank = row
            payload = payload_data if isinstance(payload_data, dict) else json.loads(payload_data)
            
            emb_vector = []
            if embedding_text and isinstance(embedding_text, str):
                emb_vector = [float(x) for x in embedding_text.strip('[]').split(',') if x.strip()]
                
            # Normalize ts_rank using rank / (rank + 1)
            raw_rank = float(rank) if rank is not None else 0.0
            normalized_lexical = raw_rank / (raw_rank + 1.0)
            
            if chunk_id in merged_candidates:
                merged_candidates[chunk_id]["lexical_score"] = normalized_lexical
                if emb_vector and not merged_candidates[chunk_id]["embedding"]:
                    merged_candidates[chunk_id]["embedding"] = emb_vector
            else:
                merged_candidates[chunk_id] = {
                    "payload": payload,
                    "embedding": emb_vector,
                    "vector_score": 0.0,
                    "lexical_score": normalized_lexical
                }

        # Calculate contributions
        v_only = 0
        l_only = 0
        both = 0
        for chunk_id, info in merged_candidates.items():
            if info["vector_score"] > 0.0 and info["lexical_score"] > 0.0:
                both += 1
            elif info["vector_score"] > 0.0:
                v_only += 1
            elif info["lexical_score"] > 0.0:
                l_only += 1
        
        self.last_query_diagnostics["hybrid_contribution"] = {
            "vector_only": v_only,
            "lexical_only": l_only,
            "both": both,
            "total": len(merged_candidates)
        }

        # 4. Calculate Hybrid Score and apply Metadata Boosting
        query_lower = query.lower()
        final_candidates = []

        for chunk_id, info in merged_candidates.items():
            payload = info["payload"]
            v_score = info["vector_score"]
            l_score = info["lexical_score"]

            # Calculate base hybrid score
            hybrid_score = (VECTOR_WEIGHT * v_score) + (LEXICAL_WEIGHT * l_score)

            # Metadata Boosting
            boost = 0.0
            
            # Boost 1: Company Match
            comp_id = payload.get("company_id") or payload.get("company_name")
            if comp_id and comp_id.lower() in query_lower:
                boost += BOOST_COMPANY

            # Boost 2: Document Type Match
            doc_type = payload.get("document_type") or payload.get("doc_type")
            if doc_type and doc_type.lower() in query_lower:
                boost += BOOST_DOCUMENT

            # Boost 3: Topic Match
            topic = payload.get("topic") or payload.get("section_heading")
            if topic and topic.lower() in query_lower:
                boost += BOOST_TOPIC

            # Boost 4: Quarter / FY Match
            quarters = ["q1", "q2", "q3", "q4", "fy"]
            payload_str = str(payload).lower()
            for qtr in quarters:
                if qtr in query_lower and qtr in payload_str:
                    boost += BOOST_QUARTER
                    break

            # Boost 5: Section Match
            section = payload.get("section") or payload.get("heading")
            if section and section.lower() in query_lower:
                boost += BOOST_SECTION

            final_score = min(1.0, hybrid_score + boost)

            final_candidates.append((chunk_id, payload, final_score, info["embedding"]))

        # Sort candidates by final score descending
        final_candidates.sort(key=lambda x: x[2], reverse=True)
        return final_candidates

    def should_expand_chunk(self, text: str) -> bool:
        """
        Checks if a chunk starts or ends mid-sentence or mid-paragraph.
        Returns True if expansion is needed, False if self-contained.
        """
        stripped = text.strip()
        if not stripped:
            return False
            
        # Starts mid-sentence if first char is lowercase
        starts_mid = stripped[0].islower()
        
        # Ends mid-sentence if last char is not sentence-ending punctuation
        ends_mid = stripped[-1] not in {'.', '?', '!', '"', ')', ']', "'"}
        
        return starts_mid or ends_mid

    def expand_adjacent_chunks(self, evidence_list: list) -> list:
        """
        Post-rerank, post-validation context expansion.
        Query N-1 and N+1 sequence neighbors using payload->>'chunk_index'
        restricted to the same document and same section.
        Deduplicates overlapping segments by index set union.
        """
        if not evidence_list:
            return evidence_list
            
        import collections
        # Group evidence list by (db_company_id, doc_id, section)
        groups = collections.defaultdict(list)
        for ev in evidence_list:
            comp_id = getattr(ev, "db_company_id", None)
            doc_id = getattr(ev, "doc_id", None)
            sec = getattr(ev, "section", "")
            chunk_idx = getattr(ev, "chunk_index", None)
            
            if comp_id and doc_id and chunk_idx is not None:
                groups[(comp_id, doc_id, sec)].append(ev)
                
        if not groups:
            return evidence_list
            
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()
        
        try:
            for (comp_id, doc_id, sec), evs in groups.items():
                target_indices = set()
                
                # Check each chunk to decide if we should expand
                for ev in evs:
                    n = ev.chunk_index
                    if self.should_expand_chunk(ev.text):
                        target_indices.update([n - 1, n, n + 1])
                    else:
                        target_indices.add(n)
                        
                # Filter out negative index values
                target_indices = {idx for idx in target_indices if idx >= 0}
                if not target_indices:
                    continue
                    
                # Fetch text for all targets
                sql = """
                    SELECT (payload->>'chunk_index')::int AS idx, payload->>'text'
                    FROM chunks
                    WHERE company_id = %s
                      AND (payload->>'doc_id' = %s OR payload->>'document_id' = %s)
                      AND coalesce(payload->>'section', payload->>'heading', payload->>'section_heading', '') = %s
                      AND (payload->>'chunk_index')::int IN %s
                    ORDER BY idx ASC;
                """
                indices_tuple = tuple(sorted(list(target_indices)))
                if len(indices_tuple) == 1:
                    cursor.execute(sql.replace("IN %s", "= %s"), (comp_id, doc_id, doc_id, sec, indices_tuple[0]))
                else:
                    cursor.execute(sql, (comp_id, doc_id, doc_id, sec, indices_tuple))
                    
                rows = cursor.fetchall()
                
                # Stitch contiguous blocks and add gap markers [...]
                stitched_paragraphs = []
                prev_idx = None
                for idx, text in rows:
                    if text and text.strip():
                        if prev_idx is not None and idx - prev_idx > 1:
                            stitched_paragraphs.append("[...]")
                        stitched_paragraphs.append(text.strip())
                        prev_idx = idx
                        
                stitched_context = "\n\n".join(stitched_paragraphs)
                
                # Deduplicate after expansion: assign stitched context to the highest scoring chunk, clear others
                if stitched_context:
                    evs.sort(key=lambda x: x.retrieval_score, reverse=True)
                    evs[0].text = stitched_context
                    for ev in evs[1:]:
                        ev.text = ""
                        
        except Exception as e:
            print(f"Error executing adjacent chunk expansion: {e}")
        finally:
            cursor.close()
            conn.close()
            
        # Return evidence with non-empty text
        return [ev for ev in evidence_list if ev.text]

    def detect_company_id(self, query: str):
        q = query.lower()
        if "bajaj" in q:
            return "bajaj_finance"
        if "pine" in q:
            return "pine_labs"
        if "hscl" in q or "himadri" in q:
            return "hscl"
        if "azad" in q:
            return "azad_engineering"
        if "maharashtra" in q:
            return "bank_of_maharashtra"
        if "ratnaveer" in q or "ratanveer" in q:
            return "ratnaveer"
        if "ather" in q:
            return "ather"
        return None

    def detect_intent(self, query: str):
        q = query.lower()
        intents = []
        if any(k in q for k in ["income", "revenue", "financial", "profit", "ebitda", "margin", "cogs", "cost", "operating", "leverage"]):
            intents.append("financials")
        if any(k in q for k in ["product", "portfolio", "rizta", "450", "platform", "halo", "scooter"]):
            intents.append("product")
        if any(k in q for k in ["growth", "growth driver", "future", "forecast", "expansion", "capacity", "investment", "launch"]):
            intents.append("future growth")
        if any(k in q for k in ["market share", "dominance", "market dominance", "vahan", "telangana", "volume", "units sold"]):
            intents.append("market share")
        if any(k in q for k in ["advantage", "competitive", "growth driver", "vertical", "brand", "patent", "r&d", "research"]):
            intents.append("competitive advantage")
        if any(k in q for k in ["store", "retail", "distribution", "dealer", "network", "experience center"]):
            intents.append("distribution")
        if any(k in q for k in ["overview", "history", "incorporation", "timeline", "milestones", "business overview", "about", "profile", "manufactur", "certif", "customer", "evolution", "presence", "key developments", "achievements"]):
            intents.append("overview")
        if any(k in q for k in ["governance", "legal", "compliance", "board of directors", "audit", "remuneration", "agm", "corporate governance"]):
            intents.append("governance")
        return intents or ["general"]

    def classify_theme(self, text: str, heading: str) -> str:
        t = (text + " " + heading).lower()
        
        # 1. Financial Statements
        if "balance sheet" in t or "statement of profit" in t or "cash flow statement" in t or "standalone financial statements" in t:
            return "financial_statements"
            
        # 2. Governance / Legal
        if "governance" in t or "auditor's report" in t or "directors" in t or "remuneration" in t or "board of" in t or "secretarial audit" in t:
            return "governance_legal"
            
        # 3. Company History / Timeline
        if "founded in" in t or "about us" in t or "our history" in t or "milestones" in t or "timeline" in t or "history of the company" in t:
            return "company_history"
            
        # 4. Snapshot
        if "snapshot" in t or "at a glance" in t or "key metrics" in t:
            return "company_snapshot"
            
        # 5. Manufacturing Capabilities
        if "manufacturing facilities" in t or "shop floor" in t or "machinery" in t or "lean manufacturing facility" in t or "sq. mts" in t or "capacity expansion" in t or "manufacturing area" in t:
            return "manufacturing_capabilities"
            
        # 6. Certifications
        if "certification" in t or "appreciation certificate" in t or "accreditation" in t or "certificate of" in t:
            return "certifications"
            
        # 7. Global Presence
        if "global footprint" in t or "global presence" in t or "sales in 12 countries" in t or "exports accounted for" in t or "international footprint" in t:
            return "global_presence"
            
        # 8. Product Portfolio / Industries
        if "product category" in t or "products end-use" in t or "portfolio of diversified products" in t or "turbine manufacturing" in t:
            return "product_portfolio"
            
        return "general"

    def classify_chunk_themes(self, text: str, heading: str) -> list:
        t = (text + " " + heading).lower()
        themes = []
        theme_keywords = {
            "Company Overview": ["overview", "about us", "profile", "about the company", "snapshot", "at a glance"],
            "History": ["history", "founded in", "incorporation", "established in", "our journey"],
            "Timeline": ["timeline", "milestones", "evolution", "key developments", "milestone"],
            "Business": ["business overview", "sectors", "industries", "business segments", "market landscape"],
            "Products": ["product", "portfolio", "turbine", "parts", "components", "product category"],
            "Manufacturing": ["manufacturing", "facilities", "plant", "machinery", "shop floor", "capacity", "manufacturing area", "sq. mts"],
            "Customers": ["customer", "client", "oem", "supplier to", "partner to"],
            "Global Presence": ["global presence", "export", "global footprint", "countries", "international", "global footprint"],
            "Certifications": ["certification", "accreditation", "qualified process", "appreciation certificate", "quality standards"],
            "Strategy": ["strategy", "growth driver", "future", "developments", "expansion", "strategic", "outlook"]
        }
        for theme, keywords in theme_keywords.items():
            if any(k in t for k in keywords):
                themes.append(theme)
        return themes or ["General"]

    def identify_requested_themes(self, query: str) -> list:
        q = query.lower()
        requested = []
        target_themes = [
            "Company Overview", "History", "Timeline", "Business", "Products",
            "Manufacturing", "Customers", "Global Presence", "Certifications", "Strategy"
        ]
        theme_keywords = {
            "Company Overview": ["overview", "about us", "profile", "about the company", "snapshot", "at a glance"],
            "History": ["history", "founded in", "incorporation", "established in", "our journey"],
            "Timeline": ["timeline", "milestones", "evolution", "key developments", "milestone"],
            "Business": ["business overview", "sectors", "industries", "business segments", "market landscape"],
            "Products": ["product", "portfolio", "turbine", "parts", "components", "product category"],
            "Manufacturing": ["manufacturing", "facilities", "plant", "machinery", "shop floor", "capacity", "manufacturing area", "sq. mts"],
            "Customers": ["customer", "client", "oem", "supplier to", "partner to"],
            "Global Presence": ["global presence", "export", "global footprint", "countries", "international", "global footprint"],
            "Certifications": ["certification", "accreditation", "qualified process", "appreciation certificate", "quality standards"],
            "Strategy": ["strategy", "growth driver", "future", "developments", "expansion", "strategic", "outlook"]
        }
        for theme in target_themes:
            if any(k in q for k in theme_keywords[theme]):
                requested.append(theme)
                
        # If it's a general overview query but none of the specific keywords matched,
        # default to all themes to ensure broad coverage
        if not requested and any(k in q for k in ["overview", "about", "snapshot", "summary"]):
            requested = target_themes
            
        return requested

    def search(self, query: str, top_k: int = 20, initial_fetch: int = 80, threshold: float = 0.50, company_id: str = None):

        # Calculate adaptive query-length threshold
        words = len(query.split())
        if words <= SHORT_QUERY_MAX_WORDS:
            actual_threshold = THRESHOLD_SHORT_QUERY
        elif words <= MEDIUM_QUERY_MAX_WORDS:
            actual_threshold = THRESHOLD_MEDIUM_QUERY
        else:
            actual_threshold = THRESHOLD_LONG_QUERY
            
        # Respect user threshold override if not the default 0.50
        if threshold != 0.50:
            actual_threshold = threshold

        # Dynamically set initial candidate fetch size
        if not company_id:
            company_id = self.detect_company_id(query)
        if company_id:
            # Increase candidate pool to capture targets under embedding mismatch
            initial_fetch = max(350, initial_fetch)
            print(f"\nSearching PostgreSQL via PGVector (fetching top {initial_fetch} candidates with company filter: {company_id})...")
        else:
            initial_fetch = max(150, initial_fetch)
            print(f"\nSearching PostgreSQL via PGVector (fetching top {initial_fetch} candidates)...")

        # Check if query decomposition is needed (broad query)
        from retrieval.sub_retrieval import decompose_and_retrieve
        decomposed_evidence = decompose_and_retrieve(self, query, company_id, threshold=actual_threshold)

        if decomposed_evidence is not None:
            # For broad queries, perform context compression and return directly,
            # bypassing global reranking, coverage selection, and slicing down to 3.
            print(f"Query Decomposition returned {len(decomposed_evidence)} unique validated chunks.")
            
            # Context Compression
            query_words = set(w.strip("?,.:;!\"'") for w in query.lower().split() if len(w) > 2)
            query_stop_words = {"what", "where", "when", "how", "who", "which", "why", "the", "and", "for", "include", "identify", "explain", "expected", "company", "position", "positioning", "latest"}
            key_concepts = query_words - query_stop_words
            
            for ev in decomposed_evidence:
                parts = ev.text.split("\n\n", 6)
                if len(parts) >= 7:
                    header = "\n\n".join(parts[:6])
                    body = parts[6]
                    
                    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
                    relevant_paragraphs = []
                    
                    for p in paragraphs:
                        p_words = set(p.lower().split())
                        if any(concept in cw for concept in key_concepts for cw in p_words) or len(p_words) < 20 or "$" in p or "INR" in p or "%" in p:
                            relevant_paragraphs.append(p)
                            
                    if relevant_paragraphs:
                        compressed_body = "\n\n".join(relevant_paragraphs)
                        ev.text = header + "\n\n" + compressed_body
                        
            return decomposed_evidence

        # If decomposed_evidence is None, perform standard single-query vector search:
        print("Query is simple. Performing standard vector search...")
        
        # Convert user query into embedding
        print("Loading Embedding Model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

        query_embedding = embedding_model.encode(query).tolist()

        del embedding_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Execute production-grade modular hybrid query
        rows = self.execute_hybrid_query(query, company_id, query_embedding, limit=initial_fetch)

        # Pre-filter candidates before CrossEncoder reranking
        from config import RERANK_LIMIT
        rows_to_process = rows[:RERANK_LIMIT]

        evidence_list = []
        for chunk_id, payload, similarity, emb_vector in rows_to_process:

            # Dynamic field mapping
            company = payload.get("company_name") or payload.get("company_id") or "Ather Energy"
            document = payload.get("document_type") or payload.get("doc_type") or "Investor Presentation"
            source_type = payload.get("chunk_type") or payload.get("source_type") or "unknown"
            page = payload.get("page_number") or payload.get("page") or 0
            text_body = payload.get("text") or payload.get("page_content") or ""
            
            heading = payload.get("heading") or payload.get("section_heading") or payload.get("section") or "general"
            topic = payload.get("topic") or payload.get("section_heading") or "general"
            document_scope = payload.get("document_scope") or "current"

            # Clean formatting for ID strings
            friendly_company = company.replace("_", " ").title() if isinstance(company, str) else str(company)
            friendly_document = document.replace("_", " ").title() if isinstance(document, str) else str(document)

            evidence = Evidence(
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
            evidence.topic = topic
            evidence.document_scope = document_scope
            evidence.db_company_id = payload.get("company_id") or company_id
            evidence.doc_id = payload.get("doc_id") or payload.get("document_id")
            evidence.chunk_index = payload.get("chunk_index")
            evidence.section = payload.get("section") or payload.get("heading") or payload.get("section_heading") or ""

            evidence_list.append(evidence)

        if not evidence_list:
            return []

        # -------------------------------------------------
        # Rerank retrieved candidates using CrossEncoder
        # -------------------------------------------------
        print(f"Reranking {len(evidence_list)} candidates using {RERANKER_MODEL}...")
        pairs = [[query, ev.text] for ev in evidence_list]
        
        import time
        # Load CrossEncoder locally
        t_r0 = time.time()
        reranker_model = CrossEncoder(RERANKER_MODEL)
        rerank_scores = reranker_model.predict(pairs, activation_fn=lambda x: x)
        t_r1 = time.time()
        self.last_query_diagnostics["reranking_latency"] = t_r1 - t_r0
        
        # Delete CrossEncoder & clean memory immediately after prediction
        del reranker_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to float list and apply intent-based boost + temperature sigmoid
        raw_scores = [float(s) for s in rerank_scores]
        detected_intents = self.detect_intent(query)
        print(f"Detected Query Intents: {detected_intents}")

        # Assign boosted and normalized score in range [0, 1]
        for ev, score in zip(evidence_list, raw_scores):
            boosted_score = score
            theme = self.classify_theme(ev.text, ev.document + " " + ev.topic)
            
            # Boost chunks whose metadata matches query intent
            if "future growth" in detected_intents and ev.document_scope == "forward_looking":
                boosted_score += 1.2
            elif "future growth" in detected_intents and ev.topic == "strategy":
                boosted_score += 0.8
                
            if "financials" in detected_intents and ev.topic == "financials":
                boosted_score += 0.8
                
            if "product" in detected_intents and ev.topic == "product":
                boosted_score += 0.8

            # Overview Query Intent filtering (boost overview targets, penalize details/noise)
            if "overview" in detected_intents:
                if theme in {"company_history", "company_snapshot"}:
                    boosted_score += 2.0
                elif theme in {"manufacturing_capabilities", "certifications", "global_presence"}:
                    boosted_score += 1.5
                elif theme == "product_portfolio":
                    boosted_score += 1.2

                # Direct target boost for corporate info and board registry pages
                text_lower = ev.text.lower()
                if "contact" in text_lower or "investor relations" in text_lower or "strategic growth advisors" in text_lower:
                    boosted_score -= 4.0

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

                # Timeline & Journey Boost
                if any(k in text_lower for k in ["highlights of our journey", "journey", "milestones", "milestone", "chronology", "timeline"]):
                    if any(q_word in query.lower() for q_word in ["timeline", "milestone", "journey", "evolution", "history", "ipo"]):
                        boosted_score += 5.0

                # Subsidiaries & Corporate Structure Boost
                if any(k in text_lower for k in ["subsidiary", "subsidiaries", "wholly owned"]):
                    if any(q_word in query.lower() for q_word in ["subsidiary", "subsidiaries", "international", "office", "acquisitions", "sharjah"]):
                        boosted_score += 4.5

                # Production Volumes & SKUs Boost
                if any(k in text_lower for k in ["processed", "production volume", "metric tonnes", "sku", "skus"]):
                    if any(q_word in query.lower() for q_word in ["processed", "capacity", "volume", "scale", "sku", "skus"]):
                        boosted_score += 4.5

            # General penalty for detailed accounting policies & notes
            text_lower = ev.text.lower()
            if any(k in text_lower for k in ["notes forming part of", "accounting policies", "basis of preparation", "accounting standards", "statement of compliance"]):
                if not any(k in text_lower for k in ["registered office", "company information"]):
                    if not any(k in query.lower() for k in ["accounting", "notes", "policies", "basis of preparation", "compliance"]):
                        boosted_score -= 4.5

            # Penalize known cover/disclaimer/signature noise pages
            if ev.page in {1, 2, 20} or "disclaimer" in ev.text.lower() or "puja aggarwal" in ev.text.lower():
                boosted_score -= 3.0

            try:
                clamped_score = max(-50.0, min(50.0, boosted_score))
                sigmoid_score = 1.0 / (1.0 + math.exp(-(clamped_score + 7.2) / 2.0))
            except Exception:
                sigmoid_score = float(boosted_score)
            ev.retrieval_score = sigmoid_score

        # Sort evidence items by their boosted reranked scores in descending order
        evidence_list.sort(key=lambda x: x.retrieval_score, reverse=True)

        # Apply similarity threshold filtering (default 0.50)
        print(f"Filtering with similarity threshold >= {actual_threshold}...")
        filtered_evidence = [ev for ev in evidence_list if ev.retrieval_score >= actual_threshold]

        # Enforce coverage-aware semantic diversity, redundancy checks, and theme coverage
        print("Applying coverage-aware selection and theme coverage satisfaction...")
        requested_themes = self.identify_requested_themes(query)
        print(f"Requested Themes for Coverage: {requested_themes}")

        diverse_evidence = []
        selected_embeddings = []
        seen_pages = set()
        covered_themes = set()

        # Set limits based on query complexity
        is_complex = len(detected_intents) > 1 or any(i in detected_intents for i in ["future growth", "competitive advantage", "overview"])
        min_chunks = 3 if is_complex else 2
        max_chunks = 6 if is_complex else 3

        # Pre-classify themes for all candidates in filtered_evidence
        evidence_with_themes = []
        for ev in filtered_evidence:
            themes = self.classify_chunk_themes(ev.text, ev.topic + " " + ev.document)
            evidence_with_themes.append((ev, themes))

        # PASS 1: Coverage Satisfaction
        # Select the highest-scoring non-redundant chunks that satisfy uncovered requested themes
        needs_model = any(not getattr(ev, "embedding", None) for ev in filtered_evidence)
        embedding_model = None
        if needs_model:
            print("Loading Embedding Model for redundancy checks...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        try:
            for ev, themes in evidence_with_themes:
                if len(diverse_evidence) >= max_chunks:
                    break

                # 1. Page-level & Document-level diversity check
                page_key = (ev.document, ev.page)
                if page_key in seen_pages:
                    continue

                # Check if this chunk covers any requested themes that are not yet covered
                uncovered_matching = [t for t in themes if t in requested_themes and t not in covered_themes]
                if not uncovered_matching:
                    continue  # Skip in Pass 1 since it doesn't contribute new theme coverage

                # 2. Jaccard Word Overlap Check (Lexical Redundancy)
                is_lexically_redundant = False
                words_curr = set(ev.text.lower().split())
                for sel_ev in diverse_evidence:
                    words_sel = set(sel_ev.text.lower().split())
                    intersection = words_curr.intersection(words_sel)
                    union = words_curr.union(words_sel)
                    jaccard = len(intersection) / len(union) if union else 0.0
                    if jaccard > 0.55:
                        is_lexically_redundant = True
                        break

                if is_lexically_redundant:
                    continue

                # 3. Vector-based Semantic Redundancy Check (Cosine Similarity)
                if getattr(ev, "embedding", None) and len(ev.embedding) > 0:
                    ev_embedding = np.array(ev.embedding)
                else:
                    ev_embedding = embedding_model.encode(ev.text) if embedding_model else None
                is_semantically_redundant = False
                if ev_embedding is not None:
                    for sel_emb in selected_embeddings:
                        sim = np.dot(ev_embedding, sel_emb) / (np.linalg.norm(ev_embedding) * np.linalg.norm(sel_emb))
                        if sim > 0.83:
                            is_semantically_redundant = True
                            break

                if is_semantically_redundant:
                    continue

                # Select this chunk to satisfy uncovered themes!
                diverse_evidence.append(ev)
                selected_embeddings.append(ev_embedding)
                seen_pages.add(page_key)
                for t in uncovered_matching:
                    covered_themes.add(t)
                print(f"Pass 1: Selected page {ev.page} ({ev.document}) for theme(s): {uncovered_matching}. Covered so far: {list(covered_themes)}")

            # PASS 2: Additional context fill
            # Only after coverage is satisfied (or candidate pool exhausted), fill remaining slots up to max_chunks
            if len(diverse_evidence) < max_chunks:
                for ev, themes in evidence_with_themes:
                    if len(diverse_evidence) >= max_chunks:
                        break

                    page_key = (ev.document, ev.page)
                    if page_key in seen_pages:
                        continue

                    # 2. Jaccard Word Overlap Check (Lexical Redundancy)
                    is_lexically_redundant = False
                    words_curr = set(ev.text.lower().split())
                    for sel_ev in diverse_evidence:
                        words_sel = set(sel_ev.text.lower().split())
                        intersection = words_curr.intersection(words_sel)
                        union = words_curr.union(words_sel)
                        jaccard = len(intersection) / len(union) if union else 0.0
                        if jaccard > 0.55:
                            is_lexically_redundant = True
                            break

                    if is_lexically_redundant:
                        continue

                    # 3. Vector-based Semantic Redundancy Check (Cosine Similarity)
                    if getattr(ev, "embedding", None) and len(ev.embedding) > 0:
                        ev_embedding = np.array(ev.embedding)
                    else:
                        ev_embedding = embedding_model.encode(ev.text) if embedding_model else None
                    is_semantically_redundant = False
                    if ev_embedding is not None:
                        for sel_emb in selected_embeddings:
                            sim = np.dot(ev_embedding, sel_emb) / (np.linalg.norm(ev_embedding) * np.linalg.norm(sel_emb))
                            if sim > 0.83:
                                is_semantically_redundant = True
                                break

                    if is_semantically_redundant:
                        continue

                    # 5. Score Gap Early Stopping Check (Only for additional context fill)
                    if len(diverse_evidence) >= min_chunks:
                        prev_score = diverse_evidence[-1].retrieval_score
                        curr_score = ev.retrieval_score
                        if prev_score - curr_score > 0.12:
                            print(f"Stopping Pass 2 early: Score gap of {prev_score - curr_score:.3f} detected between Result {len(diverse_evidence)} and candidate.")
                            break

                    # Select the chunk!
                    diverse_evidence.append(ev)
                    selected_embeddings.append(ev_embedding)
                    seen_pages.add(page_key)
                    print(f"Pass 2: Selected page {ev.page} ({ev.document}) for additional context. Theme(s): {themes}")
        finally:
            if embedding_model:
                del embedding_model
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        kept_evidence = diverse_evidence

        # -------------------------------------------------
        # Answer Sufficiency Check
        # -------------------------------------------------
        query_words = set(w.strip("?,.:;!\"'") for w in query.lower().split() if len(w) > 2)
        query_stop_words = {"what", "where", "when", "how", "who", "which", "why", "the", "and", "for", "include", "identify", "explain", "expected", "company", "position", "positioning", "latest"}
        key_concepts = query_words - query_stop_words

        covered_concepts = set()
        sufficient_evidence = []
        
        for ev in kept_evidence:
            sufficient_evidence.append(ev)
            chunk_words = set(ev.text.lower().split())
            
            # Match stems/substrings of concept terms in chunk text
            for concept in list(key_concepts - covered_concepts):
                if any(concept in cw for cw in chunk_words):
                    covered_concepts.add(concept)
            
            # If concept coverage is >= 85% and min_chunks is met, stop early
            if len(key_concepts) > 0 and len(covered_concepts) >= len(key_concepts) * 0.85 and len(sufficient_evidence) >= min_chunks:
                print(f"Stopping early: Answer sufficiency check satisfied with {len(sufficient_evidence)} chunks covering {len(covered_concepts)}/{len(key_concepts)} concept terms.")
                break

        # -------------------------------------------------
        # Context Compression
        # -------------------------------------------------
        for ev in sufficient_evidence:
            parts = ev.text.split("\n\n", 6)
            if len(parts) >= 7:
                header = "\n\n".join(parts[:6])
                body = parts[6]
                
                # Split body into paragraphs
                paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
                relevant_paragraphs = []
                
                for p in paragraphs:
                    p_words = set(p.lower().split())
                    # Keep paragraph if it matches query concepts, contains financial numbers, or is very brief
                    if any(concept in cw for concept in key_concepts for cw in p_words) or len(p_words) < 20 or "$" in p or "INR" in p or "%" in p:
                        relevant_paragraphs.append(p)
                
                if relevant_paragraphs:
                    compressed_body = "\n\n".join(relevant_paragraphs)
                    ev.text = header + "\n\n" + compressed_body

        # -------------------------------------------------
        # Evidence Validation & Coverage Reporting
        # -------------------------------------------------
        from evidence.validator import EvidenceValidator
        from reports.coverage_reporter import CoverageReporter

        validator = EvidenceValidator()
        valid_evidence = validator.filter_valid(query, sufficient_evidence)

        # Slice to restrict the final output based on query complexity
        final_evidence = valid_evidence[:max_chunks]

        # Adjacent Chunk Expansion (Post-rerank, post-validation, post-coverage)
        from config import ENABLE_ADJACENT_EXPANSION
        if ENABLE_ADJACENT_EXPANSION:
            final_evidence = self.expand_adjacent_chunks(final_evidence)

        if requested_themes:
            reporter = CoverageReporter()
            report_content = reporter.generate_report(query, requested_themes, final_evidence, self)
            reporter.save_report_to_file(query, report_content)

        print(f"Retrieved {len(final_evidence)} unique validated slide chunks after reranking, gap detection, & sufficiency check.")

        return final_evidence
