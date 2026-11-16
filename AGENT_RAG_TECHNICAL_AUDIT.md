# AGENT_RAG: Complete Technical Audit & System Architecture Report

**Document Version:** 1.0.0  
**Audit Date:** November 2026  
**Auditor:** Antigravity AI Engine  
**Workspace:** `C:\Users\komal\AGENT_RAG`  
**Git Remote:** `https://github.com/KomalBhende2/Agent_rag.git`

---

## EXECUTIVE SUMMARY

**AGENT_RAG** is an institutional financial document analysis and equity research generation platform. It transforms unstructured, visually complex corporate filings (annual reports, DRHP prospectuses, quarterly investor presentations, and earnings call transcripts) into structured, verifiable brokerage research reports.

### Key Audit Findings
1. **Pipeline Nature vs. Agent Branding**: Despite the project name and the directory `agents/`, the system does **not** employ autonomous multi-agent frameworks (such as LangGraph, CrewAI, AutoGen, or autonomous LLM tool-calling loops). The class `ResearchOrchestrator` in `agents/orchestrator.py` is an unexecuted 30-line print stub. The true operational engine is a deterministic, procedural sequential pipeline in `run_orchestrated_report.py`.
2. **LLM Provider Reality**: DeepSeek is cited in a single print statement in `agents/orchestrator.py`, but has **zero** integration in the source code. The active system integrates **Google Gemini** (`gemini-2.5-flash` primary, `gemini-1.5-flash` secondary fallback in `retrieval/synthesis.py`) backed by an offline, deterministic regex rule-based fallback.
3. **Partial Integration of Computational Geometry**: `ingestion/processors/geometric_solvers.py` contains mathematically sound implementations of the Hungarian bipartite matching algorithm, polar coordinate transformations, optimal transport (Wasserstein alignment), and Delaunay triangulation spectral clustering. However, **only the PolarChartExtractor is actively called by the ingestion pipeline** (`spatial_layout_engine.py:181`). The other three solvers are imported but bypassed in favor of heuristic coordinate bounding-box grouping.
4. **Data Point Taxonomy**: References in comments to a "209-datapoint list" represent an equity research analyst's domain checklist. In the code (`retrieval/synthesis.py`), the schema is implemented as **25 core financial aspects** customized across three corporate sectors (`manufacturing`, `financial_services`, `saas_platform`).
5. **Database Architecture**: `retrieval/vector_store.py` contains a standalone Qdrant vector client left over from an early proof-of-concept. The active retrieval engine (`retrieval/retriever.py`) bypasses Qdrant completely and queries a local **PostgreSQL + pgvector** database directly over `psycopg2`.

---

## 1. PROJECT OVERVIEW AND ARCHITECTURE

### 1.1 Project Objective
The objective of AGENT_RAG is to automate institutional equity research coverage. It ingests messy corporate filings (PDFs and presentation slide decks), preserves spatial layout structures (tables, financial comparisons, quarterly timelines, donut charts), indexes chunks in a hybrid search database, and synthesizes 12 standard brokerage sections with inline evidence citations.

### 1.2 Problem Statement
Standard Retrieval-Augmented Generation (RAG) pipelines fail on institutional financial filings due to structural information loss:
* **Table Flattening**: Sequential text readers destroy row-column alignments in financial statements.
* **Slide Layout Scrambling**: Multi-column slides (e.g. 3-card strategy presentations) read left-to-right across columns, mixing unrelated bullet points into incoherent text.
* **Visual Information Loss**: Radial donut charts, pie charts, and time-series bar graphs are invisible to standard OCR/text parsers.
* **Lexical & Acronym Blindspots**: Vector embeddings frequently miss exact financial tickers, plant unit numbers, or statutory disclosure headings.
* **Hallucination Risk**: Generic LLMs hallucinate financial metrics when disclosures are silent on specific aspects.

### 1.3 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION                              │
│  data/documents/*.pdf                                                  │
│       │                                                                │
│       ▼                                                                │
│  IngestionPipeline (ingestion/pipeline.py)                             │
│       ├─► 5-Page PyMuPDF Batch Slicing (temp_pdf_batches/)             │
│       ├─► IBM Docling Parsing (docling_extractor.py)                   │
│       ├─► pdfplumber Table Extraction (table_extractor.py)             │
│       ├─► PageLayoutClassifier (Canvas vs. Narrative vs. Raster)       │
│       ├─► SpatialLayoutEngine (2D Grid, Polar Charts, Timelines)       │
│       ├─► FigureProcessor & OpenCV Cropper (image_cropper.py)          │
│       ├─► Fallback 150 DPI Tesseract OCR (<25 words & 0 tables)        │
│       ├─► Deduplicator & Metadata Enricher                             │
│       └─► DocumentBuilder (ingestion/output/*.md)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CHUNKING & EMBEDDINGS                           │
│  ChunkFormatter (retrieval/chunk_formatter.py)                         │
│       ├─► 9 Specialized Financial Chunk Types (Tables, KPIs, Lists)    │
│       ├─► Contextual Metadata Header Prepended to Chunk Text           │
│       └─► Low-Information Filtering (drops <=15 word chunks)           │
│  EmbeddingGenerator (retrieval/embeddings.py)                          │
│       └─► BAAI/bge-large-en-v1.5 (1024-dim, L2-normalized)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DATABASE STORAGE (PostgreSQL)                      │
│  PostgreSQL with pgvector extension (127.0.0.1:5432/findoctest)        │
│  Table: chunks (id, company_id, payload JSONB, embedding vector(1024)) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        HYBRID RETRIEVAL ENGINE                         │
│  Retriever.search() (retrieval/retriever.py)                           │
│       ├─► Intent Classification & Financial Synonym Expansion          │
│       ├─► PGVector Dense Search (Cosine distance <=>) [Limit: 150]     │
│       ├─► PostgreSQL Full-Text Search (ts_rank_cd) [Limit: 80]         │
│       ├─► Weighted Fusion: 0.70 * Vector + 0.30 * Lexical              │
│       ├─► Cross-Encoder Reranking (BAAI/bge-reranker-base)             │
│       ├─► Hand-Tuned Domain Score Boosting & Policy Noise Penalties    │
│       ├─► Sigmoid Normalization [0, 1]                                 │
│       ├─► Multi-Pass Theme Coverage Diversity & Redundancy Filtering   │
│       ├─► Concept Sufficiency Check & Context Compression              │
│       └─► Adjacent Chunk Expansion (PostgreSQL query for chunk_idx ±1) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION & SYNTHESIS                           │
│  run_orchestrated_report.py                                            │
│       ├─► Iterates across 12 Brokerage Research Sections               │
│       ├─► CheckpointManager (reports/report_checkpoint.json)           │
│       ├─► Sector Pre-Filtering (Prunes inapplicable sections)          │
│       ├─► ContextCache.store() (Pass-by-Reference ref_xxxxxxxx)        │
│       ├─► RAGSynthesizer (retrieval/synthesis.py)                      │
│       │     ├─► Primary: Google Gemini 2.5 Flash                       │
│       │     ├─► Secondary: Google Gemini 1.5 Flash                     │
│       │     └─► Fallback: Deterministic Regex Rules Engine             │
│       └─► Compiled 12-Section Research Report -> output.txt            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. COMPLETE PROJECT STRUCTURE

### 2.1 Directory Responsibilities

| Directory | Responsibility | Implementation Reality |
| :--- | :--- | :--- |
| `agents/` | Intended multi-agent workflow controllers | **Mocked / Unused**. Contains only static print statements. |
| `data/documents/` | Ingestion source directory for raw PDF documents | Active. Houses target corporate filings (Ather, Bluestone, Apollo). |
| `data/extracted_figures/`| Storage for cropped image bounding boxes from PDFs | Active. Stores PNG images extracted via PyMuPDF/OpenCV. |
| `data/metadata/` | JSON metadata files for individual documents | Active. Stores checksums, document types, and scope metadata. |
| `data/page_cache/` | Single-page extracted PDF slices | Active cache. |
| `data/qdrant_db/` | Embedded Qdrant vector database storage | **Obsolete**. Leftover from early proof-of-concept; bypassed by PostgreSQL. |
| `evidence/` | Evidence validation and filtering | Active. Contains `validator.py` checking chunk length, score, and company alignment. |
| `extracted_output/` | Historical Markdown extraction dumps | Output artifact directory. |
| `ingestion/` | Core parsing, OCR, table extraction, and spatial engines | **Fully functional production engine**. Houses batching, Docling, and spatial parsers. |
| `models/` | Domain dataclass definitions | Active. Contains `evidence.py` (`Evidence` object). |
| `prompts/` | Intended external prompt template storage | **Empty directory**. Prompts are defined inline in Python files. |
| `registry/` | Document registration and checksum verification | Active. Computes SHA256 hashes and infers document types. |
| `reports/` | Coverage reporting and runtime checkpoints | Active. Houses `coverage_reporter.py`, `report_checkpoint.json`, and markdown reports. |
| `retrieval/` | Embeddings, database querying, search, and synthesis | **Core production engine**. PostgreSQL integration, hybrid search, caching, and Gemini. |
| `scratch/` | One-off assembly and debugging scripts | Transient development scripts for slide presentations. |
| `services/` | Intended microservice abstractions | **Empty package** (`__init__.py` only). |
| `temp_pdf_batches/` | Ephemeral 5-page PDF slices for memory management | Temporary runtime directory created and unlinked during ingestion. |
| `tests/` | Unit tests, pipeline validation, and benchmarks | Active. Contains verified unit tests, latency profilers, and evaluation scripts. |
| `utils/` | Utility scripts | Contains `statistics_uploader.py`. |

### 2.2 Major Python Files & Responsibilities

* **`run_orchestrated_report.py`**: The central driver script. Iterates sequentially through 12 institutional brokerage sections, loads checkpoints, triggers hybrid search, caches context, calls Gemini synthesis, and writes the compiled report to `output.txt`.
* **`run_rag.py`**: Standalone CLI runner for executing a single targeted query against the retriever and synthesizer.
* **`run_ingestion.py`**: Batch processing script that scans `data/documents/` and converts all PDFs into clean Markdown using `IngestionPipeline`.
* **`config.py`**: Global configuration module. Sets embedding model names, hybrid search weights (0.70/0.30), boosting values, thresholds, and contains monkeypatches for SSL bypass and sentence-transformers pooling.
* **`ingestion/pipeline.py`** (`IngestionPipeline`): Orchestrates memory-efficient PDF processing in 5-page chunks, coordinating Docling, pdfplumber, spatial layout engines, and OCR.
* **`ingestion/processors/spatial_layout_engine.py`** (`SpatialLayoutEngine`): Analyzes 2D bounding boxes to reconstruct structured slide tables, quarterly bar graph timelines, and donut charts.
* **`ingestion/processors/geometric_solvers.py`**: Houses 4 advanced computational geometry solvers (Hungarian, Polar, Optimal Transport, Delaunay).
* **`retrieval/retriever.py`** (`Retriever`): Core retrieval engine executing hybrid vector + lexical search, synonym expansion, cross-encoder reranking, coverage-aware diversity selection, and adjacent chunk expansion.
* **`retrieval/synthesis.py`** (`RAGSynthesizer`): Synthesis engine managing Google Gemini primary/secondary API calls and the deterministic rule-based offline fallback.
* **`retrieval/cache.py`** (`ContextCache`): In-memory key-value cache implementing pass-by-reference context decoupling.
* **`retrieval/state_manager.py`** (`CheckpointManager`): Serializes and deserializes execution state to `reports/report_checkpoint.json` for durable resumption.

---

## 3. INGESTION AND SPATIAL ENGINE

### 3.1 Document Ingestion Flow (`ingestion/pipeline.py`)
To prevent Out-Of-Memory (OOM) failures on 100+ page annual reports, `IngestionPipeline` enforces strict 5-page batching:
1. `_save_batch(pdf_path, start, end)` creates an isolated sub-PDF in `temp_pdf_batches/`.
2. `DoclingExtractor.extract()` processes the sub-PDF for semantic headings and reading order.
3. `DocumentParser.parse()` converts the Docling hierarchy into the internal `Document` schema.
4. `TableExtractor.extract_tables()` extracts tabular structures using `pdfplumber`.
5. `PageLayoutClassifier.classify()` inspects word and drawing density to classify the layout.
6. `SpatialLayoutEngine.process_page()` executes 2D coordinate parsing on structured canvases.
7. `FigureProcessor.process()` crops visual bounding boxes and runs OCR.
8. **Full-Slide Fallback OCR**: If a page has fewer than 25 words and zero tables, the entire slide is rendered at 150 DPI via PyMuPDF (`fitz.Pixmap`) and OCR'd using Tesseract to recover infographics.
9. `Deduplicator.merge()` reconciles Docling vs. pdfplumber duplicate tables.
10. `PageValidator.validate_page()` calculates lexical word recall between the raw PDF and extracted text.
11. `DocumentBuilder.build()` outputs a structured Markdown file in `ingestion/output/`.

### 3.2 Chunking and Metadata Architecture (`retrieval/chunk_formatter.py`)
Rather than arbitrary token splitting, `ChunkFormatter` produces 9 semantic chunk types:
* `HEADING_PARAGRAPH`: Section header + narrative paragraph.
* `LONG_PARAGRAPH`: Narrative text split at a 450-word boundary.
* `HEADING_TABLE`: Table title + Markdown table representation.
* `HEADING_FIGURE`: Figure title + extracted OCR text.
* `KPI_TABLE`: Key performance indicator cards formatted as tables.
* `BULLET_LIST` & `NUMBERED_LIST`: Structured disclosure lists.
* `CHART`: Visual chart data reconstructed from coordinates or OCR.
* `MIXED`: Multi-modal page summary.

**Contextual Metadata Header Injection**: Every chunk has an explicit metadata block prepended to its text prior to embedding (`chunk_formatter.py:79-88`):
```text
Company: Ather Energy
Document Type: Investor Presentation
Reporting Period: Q1 FY2026
Page Number: 14
Topic: Financial Performance
Document Scope: current
Section Heading: Financial Performance Snapshot
```
**Low-Information Filtering**: Chunks with $\le 15$ words that lack tabular or numeric data are dropped (`chunk_formatter.py:75`).

### 3.3 Spatial Layout Engine & Geometric Solvers

#### Implementation Reality of the Four Solvers (`geometric_solvers.py`)

| Solver | Mathematical Foundation | Actual Implementation | Use Case | Implementation Status |
| :--- | :--- | :--- | :--- | :--- |
| **HungarianTableMatcher** | Bipartite lattice matching via Kuhn-Munkres algorithm (`scipy.optimize.linear_sum_assignment`). Minimizes 2D distance cost matrix between cells and headers. | Builds a cost matrix penalized by $Y$-distance ($2.0\times$). Matches floating values to row/column lattice coordinates. Cells without matches default to `'—'`. | Borderless tables with blank or missing cells. Guarantees zero column shifting. | **ORPHANED / UNUSED IN PIPELINE**. Tested and verified in unit tests (`tests/test_geometric_solvers.py`), but never called in `spatial_layout_engine.py`. |
| **PolarChartExtractor** | Coordinate transformation from Cartesian $(x, y)$ to Polar $(r, \theta)$ relative to centroid; sorting by $\theta$. | Computes centroid $(\bar{x}, \bar{y})$. Converts tokens to $r = \sqrt{\Delta x^2 + \Delta y^2}$ and $\theta = \operatorname{atan2}(\Delta y, \Delta x)$. Orders tokens clockwise. | Circular pie and radial donut charts (e.g. revenue by segment, market share). | **FULLY INTEGRATED**. Called directly at line 181 of `spatial_layout_engine.py`. |
| **OptimalTransportBarAligner**| 1D Wasserstein metric / Earth Mover's Distance. Solves minimum weight assignment between metric values and timeline labels. | Builds 1D transportation cost matrix $|x_{\text{period}} - x_{\text{value}}|$ and solves assignment via `linear_sum_assignment`. | Pairing floating metric numbers above multi-period bar charts (e.g. Q1FY25, Q1FY26). | **ORPHANED / UNUSED IN PIPELINE**. Tested and verified in unit tests, but never called in `spatial_layout_engine.py`. |
| **DelaunayCardClusterer** | Delaunay triangulation (`scipy.spatial.Delaunay`) + Graph Laplacian spectral clustering. | Builds Delaunay graph of word centroids, filters edges exceeding distance threshold, constructs adjacency matrix, and clusters via eigenvectors. | Grouping irregular or staggered card point-clouds into coherent column cards. | **ORPHANED / UNUSED IN PIPELINE**. Tested and verified in unit tests, but never called in `spatial_layout_engine.py`. |

#### What `SpatialLayoutEngine` Actually Executes
Instead of calling the Hungarian, Optimal Transport, and Delaunay solvers, `spatial_layout_engine.py` employs manual geometric bounding-box heuristics:
* **Polar Donut Detector** (Line 181): Fully active via `PolarChartExtractor`.
* **2D Comparison Grid Detector** (Line 196): Groups row headers with $X \le 115$ and column headers with $240 \le Y \le 285$ using static coordinate clustering thresholds (`abs(xc - prev_xc) < 55`).
* **Multi-Period Bar Chart Detector** (Line 278): Scans for regex `Q[1-4]FY\d{2}`, captures numeric words directly above the axis box, and compiles Markdown tables.
* **Financial Performance Snapshot Detector** (Line 323): Groups blocks within $\Delta Y < 18$ points to align metric labels with currency amounts and YoY growth percentages.

---

## 4. RETRIEVAL ARCHITECTURE

The retrieval engine is implemented in `retrieval/retriever.py` (`Retriever`).

### 4.1 Embeddings & Storage
* **Embedding Model**: `BAAI/bge-large-en-v1.5` producing 1024-dimensional dense vectors, normalized to unit length (`normalize_embeddings=True`).
* **Database**: Local PostgreSQL with the `pgvector` extension.
* **Table Schema**:
  ```sql
  CREATE TABLE chunks (
      id TEXT PRIMARY KEY,
      company_id TEXT,
      payload JSONB,
      embedding VECTOR(1024)
  );
  ```

### 4.2 Hybrid Search Execution (`execute_hybrid_query`)
Search blends dense vector similarity with PostgreSQL full-text search:
1. **Dense Vector Search**:
   ```sql
   SELECT id, payload, embedding::text, 1 - (embedding <=> %s::vector) AS similarity
   FROM chunks
   WHERE company_id = %s
   ORDER BY embedding <=> %s::vector
   LIMIT 150;
   ```
2. **Lexical Full-Text Search (FTS)**:
   ```sql
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
   WHERE company_id = %s AND ... @@ to_tsquery('english', %s)
   ORDER BY rank DESC
   LIMIT 80;
   ```
3. **Score Blending**:
   $$S_{\text{hybrid}} = 0.70 \cdot S_{\text{vector}} + 0.30 \cdot S_{\text{lexical}}$$
   Candidates appearing in only one modality receive normalized fractional scores with metadata boosts (`BOOST_COMPANY = 0.05`, `BOOST_DOCUMENT = 0.03`).

### 4.3 Query Expansion & Domain Boosting
* **Financial Synonym Expansion**: `FINANCIAL_SYNONYMS` maps institutional terms to synonyms (e.g. `"outsourcing"` $\to$ `["job work", "sub-contracting", "third-party processing"]`, `"pricing power"` $\to$ `["premium pricing", "pass-through", "price increase"]`).
* **Optional LLM Expansion**: When enabled, queries Gemini to append 3–4 financial accounting keywords.
* **Cross-Encoder Reranking**: The top 40 candidates are reranked using `BAAI/bge-reranker-base`.
* **Hand-Tuned Score Adjustments**:
  * **+5.0 Boost**: Exact matches for governance, registered office, founders, or timelines when requested.
  * **-4.5 Penalty**: Statutory accounting policy notes and boilerplate compliance pages.
  * **-3.0 Penalty**: Cover pages, disclaimers, and signature blocks.
* **Sigmoid Normalization**: Raw logits are normalized to $[0, 1]$:
  $$S = \frac{1}{1 + e^{-(S_{\text{boosted}} + 7.2) / 2.0}}$$

### 4.4 Diversity, Coverage & Adjacent Expansion
* **Adaptive Thresholding**: Cutoffs scale dynamically based on query word count:
  * Short ($\le 5$ words): $0.50$
  * Medium ($\le 15$ words): $0.40$
  * Long ($> 15$ words): $0.30$
* **Multi-Pass Coverage Selection**:
  * **Pass 1**: Selects candidates strictly to satisfy uncovered query themes (`Company Overview`, `Manufacturing`, `Certifications`, etc.).
  * **Redundancy Pruning**: Discards chunks with lexical Jaccard overlap $>0.55$ or embedding cosine similarity $>0.83$.
  * **Pass 2**: Fills remaining slots up to maximum capacity; stops early if score gap between candidates exceeds $0.12$.
* **Adjacent Chunk Expansion (`expand_adjacent_chunks`)**:
  Post-reranking, the retriever queries PostgreSQL for chunks where `chunk_index` is $i - 1$ and $i + 1$ for the same document, stitching surrounding text around the evidence to prevent split-context truncation.

---

## 5. ORCHESTRATION AND 12-SECTION REPORT

### 5.1 Orchestrator Execution Flow (`run_orchestrated_report.py`)
Execution is procedural and sequential:
1. Load state checkpoint from `reports/report_checkpoint.json` if available.
2. Iterate through `BROKERAGE_SECTIONS` (1 to 12).
3. Check if the section was previously completed; if so, restore output from checkpoint.
4. **Sector Pre-Filtering**: Evaluates whether all mapped aspects for the section are marked `default_not_applicable` for the target sector. If inapplicable, database retrieval is skipped entirely, an explanatory disclaimer is logged, and state is saved.
5. **Auto-Healing Retry Loop**: Executes `retriever.search()` with up to 3 retry attempts and exponential backoff (5s, 10s).
6. Cache retrieved chunks in `ContextCache` and retrieve reference keys.
7. Call `RAGSynthesizer.generate_report_by_ref()`.
8. Persist completed section to `report_checkpoint.json`.
9. Enforce a 2.5-second cooldown to respect API rate limits.
10. Compile final output into `output.txt` and delete the checkpoint file.

### 5.2 The 12 Brokerage Sections
1. **Section 1 — Company Overview**: Corporate history, founders, headquarters, corporate ecosystem.
2. **Section 2 — Business Model**: B2B/B2C classification, monetization, distribution, customer stickiness.
3. **Section 3 — Segments & Revenue Mix**: Revenue streams, product segments, recurring ARR vs. transactional revenue.
4. **Section 4 — Market Position & Moat**: Market share, pricing power, patents, switching costs, barriers to entry.
5. **Section 5 — Key Inputs & Cost Drivers**: Raw materials, energy, wages, logistics, foreign exchange exposure.
6. **Section 6 — Value Chain Positioning**: Sourcing strategy, procurement contracts, vertical integration.
7. **Section 7 — Manufacturing & Supply Chain**: Production facilities, capacity, automation, asset heaviness, outsourcing.
8. **Section 8 — Customer Base & Aftermarket**: OEM relationships, customer concentration, aftermarket sales.
9. **Section 9 — Market Share & Dynamics**: Competitive positioning and market leadership trends.
10. **Section 10 — Geographic Presence**: Domestic presence, export footprint, international distribution.
11. **Section 11 — Growth Drivers**: Scalability drivers, capacity additions, product launches.
12. **Section 12 — Expansion & Capital Allocation**: Capex, joint ventures, partnerships, acquisitions, technology collaborations.

### 5.3 The Claimed "209 Data Points": Reality Check
The codebase references 209 data points in code comments (`run_orchestrated_report.py:23, 99`). However:
* In `retrieval/synthesis.py`, the schema defines **25 distinct financial aspects**.
* These 25 aspects are configured across three sector dictionaries: `ASPECTS` (manufacturing), `FINANCIAL_ASPECTS` (banking/NBFCs), and `SAAS_ASPECTS` (software platforms).
* `SECTION_ASPECTS_MAP` in `run_orchestrated_report.py` maps each of the 12 sections to these 25 aspects.
* **Audit Conclusion**: The "209 data points" represents the conceptual target of an institutional research checklist, but the actual executable schema evaluates **25 structured aspects**.

### 5.4 Synthesis & Citations (`retrieval/synthesis.py`)
`RAGSynthesizer` formats evidence as:
`[Evidence X] Company: {ev.company} | Document: {ev.document} | Page: {ev.page}`
The prompt instructs Gemini to cite `[Evidence X]` for every claim and output explicit standardized disclaimers when evidence is absent. If Gemini fails, `_generate_with_rules()` maps aspects using keyword regex, computes a percentage grounding score, and renders an ASCII progress bar report.

---

## 6. CACHING AND CHECKPOINTING

### 6.1 In-Memory Context Cache (`retrieval/cache.py`)
* **Design Pattern**: Pass-by-Reference in-memory key-value store (`ContextCache`).
* **Implementation**:
  * `store(items)` generates UUID reference keys (`f"ref_{uuid.uuid4().hex[:8]}"`), stores the evidence objects in a class-level dictionary `_cache`, and returns the keys.
  * `resolve(keys)` maps keys back to stored evidence objects.
* **Limitations**:
  * **Memory Leak Risk**: No TTL or maximum capacity limit. `_cache` persists until `ContextCache.clear()` is called.
  * **Single Process Only**: In-memory dictionary cannot be accessed across distributed worker processes.

### 6.2 Durable Checkpoint Manager (`retrieval/state_manager.py`)
* **Design Pattern**: Persistent state serialization (`CheckpointManager`).
* **Implementation**:
  * Serializes state to `reports/report_checkpoint.json` containing `company_id`, `completed_sections` (list of completed section IDs), and `section_outputs` (dictionary mapping section IDs to Markdown text).
  * `load_checkpoint()` restores state on startup. If an interrupted run is detected for the same company, completed sections are loaded from disk and skipped.
  * `clear_checkpoint()` deletes the JSON file upon successful report completion.

---

## 7. TESTING AND BENCHMARKING

### 7.1 Test Suite Audit & Execution Results

All available unit tests were executed inside the local virtual environment:

```powershell
# Geometric Solvers Test Suite
.\venv\Scripts\python.exe -m unittest tests/test_geometric_solvers.py
# Result: Ran 4 tests in 0.056s. All tests PASSED.

# Universal Pipeline Integration Test Suite
.\venv\Scripts\python.exe -m unittest tests/test_universal_pipeline.py
# Result: Ran 5 tests in 0.325s. All tests PASSED.
```

### 7.2 Detailed Test Coverage

| Test Module | Scope | Status | Observations |
| :--- | :--- | :--- | :--- |
| `tests/test_geometric_solvers.py` | Unit | **PASSED (0.056s)** | Validates empty cell preservation in Hungarian matcher, polar clockwise sorting, optimal transport bar pairing, and Delaunay card clustering. |
| `tests/test_universal_pipeline.py` | Integration | **PASSED (0.325s)** | Validates page classification and spatial layout reconstruction on test investor presentation slides (Pages 14, 15, 17, 18, 19, 20). |
| `tests/benchmark_runner.py` | Benchmark | **Executable with DB** | Measures precision, recall, and aspect coverage across standardized company queries. Requires active PostgreSQL. |
| `tests/profile_latency.py` | Profiling | **Executable with DB** | Measures latency breakdown across vector search, FTS, and reranking. Requires active PostgreSQL. |
| `tests/test_deduplicator.py` | Unit | **0 Bytes (Empty)** | Unimplemented test stub. |
| `tests/test_metadata_enricher.py` | Unit | **0 Bytes (Empty)** | Unimplemented test stub. |
| `tests/test_text_cleaner.py` | Unit | **0 Bytes (Empty)** | Unimplemented test stub. |

---

## 8. OLD VS NEW COMPARISON

### 8.1 Comparison Matrix: Committed GitHub Baseline vs. Current Local Workspace

| Dimension | GitHub Baseline (`origin/main`, Aug 4, 2026) | Current Local Workspace (Nov 2026) | Status |
| :--- | :--- | :--- | :--- |
| **Commit ID** | `09a93b2` ("Sceonf commit") | Same base commit + extensive uncommitted working tree | Diverged |
| **Raw Documents** | Committed ~100MB of raw binary PDFs (`Ather AR.pdf`, `Q1-Q4 PPTs`) into Git history | Cleaned up. Raw PDFs deleted from Git index; newer documents used locally | Improved |
| **Spatial Engine** | Basic document parsing via standard Docling/pdfplumber | Custom 2D spatial layout engine with polar coordinate chart transformations | Implemented |
| **Geometric Solvers** | None | 4 mathematical solvers implemented in `geometric_solvers.py` | Partially Integrated |
| **Retrieval Engine** | Basic dense vector top-k cosine similarity | **Hybrid Search**: PGVector (70%) + PostgreSQL FTS (30%) + Cross-Encoder reranking | Implemented |
| **Domain Boosting** | Flat scoring across all chunks | Intent-based boosting (+5.0 governance, -4.5 accounting policy penalties) | Implemented |
| **Context Expansion** | Isolated individual chunks | **Adjacent Chunk Expansion**: Stitches $i \pm 1$ neighbor chunks from PostgreSQL | Implemented |
| **Report Scope** | Single-query ad-hoc script (`run_rag.py`) | Automated 12-section institutional brokerage orchestrator (`run_orchestrated_report.py`)| Implemented |
| **State Persistence** | None (crashes lost all progress) | Persistent JSON checkpointing with auto-resume (`state_manager.py`) | Implemented |
| **Context Caching** | Direct text payload copying | Pass-by-reference key caching (`cache.py`) | Implemented |
| **Temporary Files** | Committed zip archives and pgvector DLL files | Removed obsolete binary archives from active workspace | Cleaned up |

---

## 9. TECHNICAL DEBT AND RECOMMENDATIONS

### Critical Priority
1. **Remove Global SSL Disabling**: Remove `ssl._create_default_https_context = ssl._create_unverified_context` and the monkeypatches on `requests.Session` and `httpx.Client` in `config.py`. Specify custom proxy certificates using standard environment variables (`REQUESTS_CA_BUNDLE`).
2. **Externalize Database Credentials**: Replace the hardcoded connection string `postgresql://postgres:postgre123@127.0.0.1:5432/findoctest` in `retrieval/retriever.py:82` with `os.environ.get("DATABASE_URL")`.
3. **Integrate Orphaned Geometric Solvers**: Connect `HungarianTableMatcher`, `OptimalTransportBarAligner`, and `DelaunayCardClusterer` into `spatial_layout_engine.py` to replace static bounding-box coordinate heuristics.

### High Priority
4. **Wire or Remove Stub Files**: Connect `agents/orchestrator.py` and `app.py` to the actual production pipeline (`run_orchestrated_report.py`) or remove them to eliminate misleading dead code.
5. **Reconcile Vector Store Module**: Replace the obsolete Qdrant code in `retrieval/vector_store.py` with a clean `PgVectorStore` abstraction matching the actual PostgreSQL implementation in `retriever.py`.
6. **Generalize Coordinate Thresholds**: Replace absolute pixel cutoffs (`y >= 505`, `w[2] <= 115`) and company-specific regex (`PCJ`) in `spatial_layout_engine.py` with normalized relative coordinates ($x/W, y/H$).

### Medium Priority
7. **Add Eviction to ContextCache**: Implement an LRU eviction policy (`collections.OrderedDict`) with a maximum capacity to prevent long-running memory leaks.
8. **Implement Missing Unit Tests**: Populate the 0-byte test files (`test_deduplicator.py`, `test_metadata_enricher.py`, `test_text_cleaner.py`).

---

*Report compiled and verified against codebase source files on November 16, 2026.*
