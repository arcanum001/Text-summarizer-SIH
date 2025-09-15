Below is a complete, end-to-end design that merges everything the original Document Processing Project already provides plus the new modules and customizations needed for KMRL’s real-world document workflow.
The goal is to show a single, integrated system—from ingestion to multilingual summarization and searchable output—all running fully offline in a Docker container.


---

1️⃣  System Overview

Objective:
Automatically ingest thousands of PDFs (English/Malayalam, scanned or digital), identify sections relevant to each department or role, and produce concise, traceable summaries for rapid decision-making.

High-Level Flow

Document Sources  →  Ingestion & OCR  →  Chunking & Metadata
                      ↓
               Search & Ranking (BM25 + Semantic)
                      ↓
          Persona-specific Abstractive Summaries
                      ↓
   Structured Output (JSON / DB / Dashboard + Trace Links)


---

2️⃣  Core Features Already in the Project

Stage	Existing Implementation	Role in the KMRL Solution

PDF text extraction	pdf_parser.py uses PyMuPDF to read text and split into paragraph-based chunks with metadata (file name, page number, density).	Handles native, text-based PDFs efficiently.
Information retrieval	search_engine.py combines BM25 keyword ranking with sentence-transformers/all-MiniLM-L6-v2 semantic similarity for re-ranking.	Ensures highly relevant content for queries like “urgent track maintenance notices.”
Summarization	result_generator.py uses T5-small for abstractive summaries of the top chunks.	Delivers concise, human-readable digests.
Offline, containerized execution	Docker image (python:3.9-slim) pre-installs all dependencies and downloads models during build, so runtime requires no internet.	Crucial for KMRL’s secure or air-gapped networks.
Configuration	config.json for chunking thresholds, ranking parameters, and filtering rules.	Allows tuning for different departments without code changes.
Git LFS	Tracks large model files (t5-small/) and big PDFs efficiently.	Keeps repository manageable.



---

3️⃣  New Enhancements for KMRL

Requirement	Enhancement	Implementation Details

Scanned & mixed-content PDFs	OCR Layer	Integrate pytesseract with opencv-python-headless. Preprocess images (deskew, binarize) and feed to Tesseract for text extraction.
Bilingual text (English + Malayalam)	Language Detection & Multilingual Models	Use langdetect to label chunks. Add google/mt5-small or facebook/mbart-large-50-many-to-many-mmt for Malayalam summarization. Route chunks to the correct summarizer automatically.
Tables & structured data	Table Extraction	Use camelot-py[cv] or tabula-py to extract CSV/JSON from tables before summarization, preserving key numeric data (e.g., maintenance job cards).
Department-specific insights	Persona Expansion	Create JSON/YAML persona configs: Engineering_Manager, Procurement_Officer, HR_Lead, etc., each with weighted keywords and sample intent descriptions.
Historical search & dashboards	Database + API	Store result.json into PostgreSQL or Elasticsearch (via psycopg2-binary or elasticsearch library). Provide a lightweight FastAPI endpoint for internal dashboards.
Scheduling	Automated Cron Jobs	Run nightly Docker commands (see below) for daily digests to all departments.
Traceability	Rich Metadata	Extend chunk metadata to include file path, page range, OCR confidence score, and hash of the source PDF for audit trails.



---

4️⃣  Complete Processing Pipeline

Step 1 – Ingestion

Sources: e-mails, SharePoint, Maximo exports, WhatsApp PDFs, cloud links.

Collector script: places all new PDFs into /data/kmrl/input.


Step 2 – Text & Data Extraction

1. PyMuPDF extracts text where possible.


2. OCR fallback (pytesseract) handles image-only pages.


3. Language detection tags each chunk as English, Malayalam, or mixed.


4. Table extraction with camelot produces structured CSV/JSON for numeric tables.



Output: chunks.json with metadata

{
  "filename": "Safety_Bulletin_2025_09.pdf",
  "page": 3,
  "language": "Malayalam",
  "content": "…",
  "ocr_confidence": 0.93
}

Step 3 – Ranking

BM25 retrieval finds top k chunks for each persona query.

Semantic re-ranking with all-MiniLM-L6-v2 embeddings + cosine similarity.

Combined score = 0.6 × semantic + 0.4 × BM25.


Step 4 – Summarization

Language-aware summarizer:

English → T5-small

Malayalam → mT5-small


Output: 40-150 word summaries depending on persona configuration.


Step 5 – Output & Delivery

result.json: ranked summaries with metadata and links to original file.

Optionally insert into PostgreSQL/Elasticsearch.

A simple FastAPI service can expose REST endpoints for dashboards or email digests.



---

5️⃣  Folder & File Layout

kmrl-doc-processor/
├── Dockerfile
├── requirements.txt
├── config.json
├── main.py
├── pdf_parser.py        # extended for OCR & multilingual
├── search_engine.py
├── result_generator.py  # supports mT5
├── ingestion/
│   └── collector.py
├── input/               # incoming PDFs
├── output/              # JSON & logs
└── models/
    ├── t5-small/
    └── mt5-small/


---

6️⃣  Requirements Additions

requirements.txt (incremental):

pytesseract
opencv-python-headless
langdetect
camelot-py[cv]
psycopg2-binary      # if using PostgreSQL
elasticsearch        # if using Elasticsearch
fastapi uvicorn      # optional REST API


---

7️⃣  Building & Running

# Build the image
docker build -t kmrl-doc-processor .

# Single run with a persona query
docker run --rm \
  -v /data/kmrl/input:/app/input \
  -v /data/kmrl/output:/app/output \
  kmrl-doc-processor \
  python /app/config/main.py \
  "Latest safety and regulatory updates" "Engineering_Manager"

Automated Nightly Runs

Add to crontab -e:

0 2 * * * docker run --rm \
    -v /data/kmrl/input:/app/input \
    -v /data/kmrl/output:/app/output \
    kmrl-doc-processor \
    python /app/config/main.py "Daily finance summary" "Finance_Officer"


---

8️⃣  Example Output (result.json)

{
  "persona": "Engineering_Manager",
  "generated_at": "2025-09-13T02:15:00Z",
  "summaries": [
    {
      "source_file": "Incident_Report_Sept12.pdf",
      "pages": [4,5],
      "language": "English",
      "summary": "Track section near Depot 2 reported abnormal vibration..."
    },
    {
      "source_file": "Regulatory_Circular_Sept13.pdf",
      "pages": [1],
      "language": "Malayalam",
      "summary": "മെട്രോ സുരക്ഷാ നിർദേശങ്ങളിൽ പുതിയ ചട്ടം ..."
    }
  ]
}


---

9️⃣  Benefits to KMRL

Faster decisions: Shift managers read 2-minute summaries, not 200-page PDFs.

Cross-department visibility: One pipeline feeds Engineering, HR, Procurement, etc.

Regulatory compliance: Critical circulars surfaced automatically with traceable links.

Institutional memory: Searchable JSON/DB store preserves knowledge even as staff change.

Offline security: All dependencies pre-installed in Docker; runs entirely inside KMRL’s network.



---

Final Takeaway

This integrated design keeps the strong foundation of the original project (chunking, BM25+semantic ranking, T5 summarization, offline Docker deployment) and adds OCR, multilingual support, table handling, persona-based ranking, and automated scheduling.
With these extensions, KMRL gains a production-ready, secure, multilingual document intelligence platform capable of processing every regulatory directive, maintenance report, and invoice into rapid, actionable insights.
