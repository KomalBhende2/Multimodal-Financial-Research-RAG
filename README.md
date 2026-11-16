# Multimodal Financial Research Automation | RAG-Based AI System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.7+-green.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An institutional-grade, multimodal Retrieval-Augmented Generation (RAG) platform that ingests complex corporate financial disclosures (annual reports, DRHP prospectuses, quarterly investor presentations, and earnings call transcripts), reconstructs 2D spatial layouts, and synthesizes structured 12-section equity research coverage reports with verified evidence citations.

---

## 🏛️ System Architecture

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

## ⚡ Key Engineering Features

### 1. 2D Spatial Document Engine & Geometric Solvers
* **Polar Coordinate Transformation**: Transforms radial donut charts and pie charts from Cartesian $(x, y)$ space to polar $(r, \theta)$ space, sorting tokens clockwise to reconstruct exact slice breakdowns into Markdown tables.
* **Multi-Period Bar Chart Alignment**: Identifies quarterly timeline labels (e.g., `Q1FY25`, `Q1FY26`) and aligns floating metric values above bars into clean time-series tables.
* **Memory-Optimized 5-Page Batching**: PyMuPDF slices 100+ page annual reports into temporary 5-page sub-documents to prevent IBM Docling and OCR from causing Out-of-Memory (OOM) crashes.
* **Infographic Fallback OCR**: Pages with $<25$ words and zero tables are automatically rasterized at 150 DPI and OCR'd via Tesseract to eliminate visual information loss.

### 2. Hybrid Retrieval with Intent-Based Reranking
* **Weighted Hybrid Search**: Fuses dense vector similarity (`BAAI/bge-large-en-v1.5`, 1024-dim) with lexical full-text search (`ts_rank_cd` over PostgreSQL tsvectors) at a 70/30 weighting.
* **Financial Keyword Expansion**: Bridges vocabulary gaps with domain dictionaries (e.g., mapping `"outsourcing"` $\to$ `["job work", "sub-contracting"]`, `"recurring revenue"` $\to$ `["ARR", "MRR", "licensing"]`).
* **Cross-Encoder Reranking**: The top 40 candidates are reranked via `BAAI/bge-reranker-base`.
* **Domain Score Adjustments**: Hand-tuned score boosts (+5.0 for governance/promoter registries) and penalties (-4.5 for boilerplate accounting policy notes).
* **Adjacent Chunk Expansion**: Queries PostgreSQL for neighbor chunks ($i \pm 1$) post-reranking to restore sentences cut off by chunk boundaries.

### 3. Institutional 12-Section Report Orchestration
Automates standard sell-side equity research coverage across 12 fundamental sections:
1. **Company Overview** (Founders, history, corporate profile)
2. **Business Model** (B2B/B2C, monetization, distribution)
3. **Segments & Revenue Mix** (Product segments, recurring vs. transactional revenue)
4. **Market Position & Moat** (Market share, pricing power, entry barriers)
5. **Key Inputs & Cost Drivers** (Raw materials, energy, labor, inflation, forex)
6. **Value Chain Positioning** (Sourcing strategy, supplier concentration)
7. **Manufacturing & Supply Chain** (Capacity, automation, asset heaviness)
8. **Customer Base & Aftermarket** (OEM relationships, customer stickiness)
9. **Market Share & Dynamics** (Competitive dynamics, industry ranking)
10. **Geographic Presence** (Domestic footprint, export reach)
11. **Growth Drivers** (Scalability drivers, capacity additions, new product launches)
12. **Expansion & Capital Allocation** (Capex, joint ventures, partnerships, M&A)

### 4. Fault Tolerance & Memory Optimization
* **Pass-by-Reference Context Caching (`ContextCache`)**: Stores retrieved chunks in an in-memory key-value store, passing lightweight keys (`ref_xxxxxxxx`) to save RAM.
* **Durable Checkpointing (`CheckpointManager`)**: Progress is serialized to `reports/report_checkpoint.json` after every section. If a run is interrupted by network timeouts or API rate limits, it automatically resumes where it left off.
* **Deterministic Offline Fallback**: If LLM API keys are unavailable or fail, the system falls back to a deterministic regex rules engine with percentage grounding scores and ASCII progress tracking.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Environment** | Python 3.12+, Windows / Linux / macOS |
| **Document Ingestion** | PyMuPDF (`fitz`), IBM Docling, `pdfplumber`, Tesseract OCR (`pytesseract`), OpenCV (`cv2`) |
| **Vector Database & FTS** | PostgreSQL 16+, `pgvector` (cosine distance `<=>`), `psycopg2` |
| **Embeddings & Reranking** | SentenceTransformers (`BAAI/bge-large-en-v1.5`), CrossEncoder (`BAAI/bge-reranker-base`) |
| **LLM Synthesis** | Google Gemini API (`gemini-2.5-flash`, `gemini-1.5-flash`) |
| **Scientific & Geometry** | NumPy, SciPy (`scipy.optimize`, `scipy.spatial`) |

---

## 📂 Project Structure

```
├── run_orchestrated_report.py   # Primary production driver: 12-section research compiler
├── run_rag.py                   # Single-query CLI runner for arbitrary research queries
├── run_ingestion.py             # Batch PDF processor converting filings to Markdown
├── config.py                    # Global weights, thresholds, model names, and paths
│
├── ingestion/                   # Ingestion and spatial layout pipeline
│   ├── pipeline.py              # 5-page batched IngestionPipeline
│   ├── extractors/              # Docling, pdfplumber, and table extractors
│   ├── ocr/                     # Tesseract OCR wrapper
│   └── processors/              # SpatialLayoutEngine, PageClassifier, GeometricSolvers
│
├── retrieval/                   # Storage, search, reranking, and synthesis
│   ├── retriever.py             # Core hybrid retriever (PGVector + FTS + Reranking)
│   ├── synthesis.py             # RAGSynthesizer (Gemini API + Rules fallback)
│   ├── embeddings.py            # SentenceTransformer multi-process encoder
│   ├── cache.py                 # Pass-by-Reference ContextCache
│   ├── state_manager.py         # CheckpointManager for durable execution
│   └── chunk_formatter.py       # 9 specialized financial chunk formats
│
├── evidence/                    # Evidence validator (length, score, company alignment)
├── models/                      # Domain dataclass definitions (Evidence, Document)
├── registry/                    # Document registry & SHA256 checksum verification
├── reports/                     # Generated coverage reports & benchmark logs
└── tests/                       # Unit tests, universal pipeline tests, benchmark runners
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.12 or higher
* PostgreSQL with the `pgvector` extension installed
* Tesseract OCR installed and added to your system PATH

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/KomalBhende2/Multimodal-Financial-Research-RAG.git
cd Multimodal-Financial-Research-RAG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/findoctest
```

### 4. Running the System

#### A. Ingest Corporate PDFs
Place target PDF filings into `data/documents/` and run:
```bash
python run_ingestion.py
```

#### B. Generate a Full 12-Section Research Report
Generate institutional coverage for any indexed company:
```bash
python run_orchestrated_report.py --company ratnaveer
```
*The completed report will be saved to `output.txt`.*

#### C. Run an Ad-Hoc Targeted Query
```bash
python run_rag.py --company ratnaveer --query "Raw materials, input costs, logistics, and supplier concentration"
```

#### D. Run the Test Suite
```bash
python -m unittest tests/test_geometric_solvers.py
python -m unittest tests/test_universal_pipeline.py
```

---

## 🔍 Technical Audit

A comprehensive line-by-line architectural audit and verification report is available in [`AGENT_RAG_TECHNICAL_AUDIT.md`](AGENT_RAG_TECHNICAL_AUDIT.md).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
