AdobeIH1b Document Processing Project
This project processes PDF documents to rank and summarize sections based on a user-provided query and persona, leveraging advanced natural language processing (NLP) techniques. It operates within a Docker container (doc-relevance-processor) with all dependencies pre-installed, ensuring offline execution. The system extracts text from PDFs in the input/ directory, ranks sections for relevance, and generates summaries, saving results as JSON files in the output/ directory. The project is designed for portability, reliability, and offline use, with large files (t5-small/, PDFs) managed via Git LFS.
Approach
The project implements a robust pipeline for document processing, combining information retrieval, semantic ranking, and text summarization. Below is a detailed breakdown of the approach:

PDF Text Extraction:

Tool: pdf_parser.py uses PyMuPDF to extract text from PDFs in input/.
Process: PDFs are split into chunks based on paragraphs (double newlines), with configurable parameters (min_chunk_word_count=50, max_heading_word_count=10) from config.json. Each chunk includes metadata (document name, section title, page number, content density).
Purpose: Ensures only meaningful text segments are processed, filtering out short or irrelevant sections.


Search and Ranking:

Tool: search_engine.py integrates BM25 (rank-bm25) for initial keyword-based ranking and sentence-transformers/all-MiniLM-L6-v2 for semantic re-ranking.
Process:
Indexing: Chunks are indexed using BM25 for keyword matching and encoded into embeddings for semantic similarity.
Query Processing: The query (e.g., "Prepare a vegetarian buffet-style dinner menu") is enhanced with persona context (e.g., "Health_Conscious") to form an "ideal document" (create_dynamic_ideal_document in main.py).
Ranking: BM25 retrieves the top top_k_retrieval=100 chunks, which are re-ranked using cosine similarity between chunk embeddings and the ideal document’s embedding. A blended score (intent_weight=0.6) combines BM25 and semantic scores.


Purpose: Delivers highly relevant chunks tailored to the query and persona, prioritizing actionable content.


Summarization:

Tool: result_generator.py uses the T5-small model (transformers) for abstractive summarization.
Process: Top-ranked chunks are summarized (max 150 words, min 40 words) to produce concise, actionable insights. Summaries are paired with original chunks and metadata in result.json.
Purpose: Condenses lengthy sections into digestible summaries, preserving key information.


Offline Execution:

Dockerfile: Based on python:3.9-slim, it pre-installs system dependencies (libmupdf-dev, etc.), Python libraries (requirements.txt), NLTK data (punkt, averaged_perceptron_tagger, wordnet), and the T5-small model (setup_t5.py).
Process: All dependencies are bundled during the Docker build, eliminating runtime internet access. The project folder is mounted as a volume for persistent input/ and output/ access.
Purpose: Ensures the container is self-contained, ideal for secure or disconnected environments.


Configuration and Extensibility:

Config: config.json specifies paths (/app/input, /app/output, /app/config/t5-small), model settings, search parameters (top_k_retrieval, top_k_rerank), and filtering rules (intent_weight, irrelevant_terms).
Synonym Expansion: synonyms.py uses NLTK’s WordNet to expand query terms, enhancing search recall.
Error Handling: Errors are logged to output/error_log.json for debugging.
Purpose: Provides flexibility to adapt the pipeline to different use cases (e.g., business forms, menus).


Git LFS Integration:

Large files (t5-small/~300 MB, PDFs in input/) are tracked with Git LFS to manage storage efficiently on GitHub.
Purpose: Keeps the repository lightweight while supporting large model and data files.



The main script (main.py) orchestrates the pipeline, accepting a query and persona as command-line arguments. Outputs include:

result.json: Ranked sections, summaries, and metadata.
chunks.json: Extracted PDF chunks with metadata.
error_log.json: Runtime errors (if any).

Models and Libraries
Models

T5-small:
Library: transformers==4.31.0
Description: A transformer-based model for abstractive text summarization, pre-trained on a diverse corpus. Stored in t5-small/ (~300 MB).
Use: Generates concise summaries of ranked document chunks.


sentence-transformers/all-MiniLM-L6-v2:
Library: sentence-transformers==2.2.2
Description: A compact sentence embedding model optimized for semantic similarity tasks.
Use: Encodes chunks and queries for semantic re-ranking.



Libraries

PyMuPDF==1.24.10: Extracts text from PDFs, supporting complex layouts and metadata.
transformers==4.31.0: Provides T5 model, tokenizer, and utilities for NLP tasks.
sentencepiece==0.1.99: Tokenization support for T5, enabling efficient text processing.
torch==2.0.1: PyTorch framework for model inference (T5, sentence-transformers).
numpy==1.24.4: Numerical computations for embedding operations and scoring.
scikit-learn==1.3.2: Implements cosine similarity for ranking chunk embeddings.
nltk==3.8.1: Supports tokenization, part-of-speech tagging, and WordNet synonym expansion.
rank-bm25==0.2.2: BM25 algorithm for keyword-based document retrieval.
sentence-transformers==2.2.2: Provides the all-MiniLM-L6-v2 model for semantic embeddings.

Setup Instructions
Prerequisites

Docker: Install from docker.com. Verify: docker --version.
Git and Git LFS:
Install Git: git-scm.com. Verify: git --version.
Install Git LFS: git-lfs.github.com. Verify: git lfs --version.
Run: git lfs install.


GitHub Repository:
Clone: git clone https://github.com/arcanum001/Adobe1b.git.
Or initialize locally and push to https://github.com/arcanum001/Adobe1b.


Input PDFs:
Place at least one .pdf in input/ (e.g., menu.pdf).
Example content: "Vegetarian Menu: Quinoa salad with chickpeas, tomatoes, cucumber, olive oil, and lemon juice."
PDFs must contain extractable text (not scanned images).


Authentication:
HTTPS: Create a Personal Access Token (PAT) at GitHub Settings > Developer settings with repo scope.
SSH: Generate an SSH key (ssh-keygen -t ed25519 -C "your_email@example.com") and add to GitHub. Verify: ssh -T git@github.com.



Project Structure
AdobeIH1b/
├── Dockerfile
├── README.md
├── requirements.txt
├── main.py
├── config.json
├── pdf_parser.py
├── setup_t5.py
├── synonyms.py
├── irrelevant_terms.json
├── relevant_terms.json
├── approach_explanation.md
├── search_engine.py
├── result_generator.py
├── input/
│   └── menu.pdf
├── output/
└── t5-small/

Docker Commands

Navigate to Project Directory (Windows):
cd C:\Users\harsh\Downloads\AdobeIH1b

Or on Unix-like systems:
cd /path/to/AdobeIH1b


Build Docker Image:
docker build -t doc-relevance-processor .


Description: Builds the doc-relevance-processor image using python:3.9-slim as the base. Installs system dependencies (libmupdf-dev, etc.), Python packages (requirements.txt), NLTK data, and T5-small model.
Output: Creates a local image named doc-relevance-processor. Check with docker images.
Time: May take 5-15 minutes depending on internet speed and system resources (downloads ~1 GB of models/dependencies).


Run Docker Container:
docker run --rm -v C:\Users\harsh\Downloads\AdobeIH1b\input:/app/input -v C:\Users\harsh\Downloads\AdobeIH1b\output:/app/output -v C:\Users\harsh\Downloads\AdobeIH1b:/app/config doc-relevance-processor python /app/config/main.py "Prepare a vegetarian buffet-style dinner menu" "Health_Conscious"


Description:
--rm: Removes the container after execution.
-v: Mounts local directories (input/, output/, project root) to container paths (/app/input, /app/output, /app/config).
Executes main.py with a sample query and persona.


Output: Generates result.json, chunks.json, and error_log.json (if errors) in output/.
Unix-like systems:docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output -v $(pwd):/app/config doc-relevance-processor python /app/config/main.py "Prepare a vegetarian buffet-style dinner menu" "Health_Conscious"




Verify Outputs:
dir C:\Users\harsh\Downloads\AdobeIH1b\output
type C:\Users\harsh\Downloads\AdobeIH1b\output\result.json


Description: Lists files in output/ and displays result.json (contains ranked sections, summaries, metadata).
Expected Files:
result.json: Structured output with sections, subsections, and metadata.
chunks.json: Raw extracted chunks with metadata.
error_log.json: Errors (if any).


Unix-like systems:ls -l output/
cat output/result.json




Debug Dependency Installation (if build fails):
docker run --rm -v C:\Users\harsh\Downloads\AdobeIH1b:/app -it doc-relevance-processor bash
cat pip_install.log
cat t5_download.log


Description: Enters the container to inspect build logs for Python packages (pip_install.log) or T5 model download (t5_download.log).
Use: Identify missing dependencies or network issues during build.


Verify Dependencies and Models:
docker run --rm -v C:\Users\harsh\Downloads\AdobeIH1b:/app -it doc-relevance-processor bash
python -c "import transformers, sentencepiece, torch, numpy, sklearn, nltk, rank_bm25, sentence_transformers, fitz; print('All dependencies imported')"
ls -l /app/config/t5-small
ls -l /usr/share/nltk_data


Description:
Tests import of all Python libraries.
Lists files in t5-small/ (T5 model) and /usr/share/nltk_data (NLTK resources).


Output: Confirms dependencies are installed and models/data are present.


Run Alternative Queries:
docker run --rm -v C:\Users\harsh\Downloads\AdobeIH1b\input:/app/input -v C:\Users\harsh\Downloads\AdobeIH1b\output:/app/output -v C:\Users\harsh\Downloads\AdobeIH1b:/app/config doc-relevance-processor python /app/config/main.py "Create a business contract template" "Business_Analyst"


Description: Runs main.py with a different query and persona to test flexibility.
Use: Validates the pipeline with diverse inputs.



Troubleshooting

No PDFs in input/:

Symptom: No chunks extracted in error_log.json.
Fix: Place a valid .pdf in input/ with extractable text (not scanned images). Example:Vegetarian Menu
Quinoa salad with chickpeas, tomatoes, cucumber, olive oil, and lemon juice.

Convert to input/menu.pdf using Word or an online tool.
Verify: dir C:\Users\harsh\Downloads\AdobeIH1b\input.


Docker Build Fails:

Symptom: Errors during pip install or T5 download.
Fix:
Check logs: cat pip_install.log or cat t5_download.log (see debug command above).
Ensure internet access during build for downloading dependencies.
Verify system dependencies (libmupdf-dev, etc.) are installed correctly.


Share: Log output for further assistance.


Missing search_engine.py or result_generator.py:

Symptom: ImportError during runtime.
Fix: Ensure files exist (dir C:\Users\harsh\Downloads\AdobeIH1b\search_engine.py). Use provided minimal implementations or share your versions.
Verify: git status to confirm files are tracked.


Git LFS Issues:

Symptom: git: 'lfs' is not a git command or File t5-small/somefile.bin exceeds GitHub's file size limit.
Fix:
Install Git LFS: git lfs install.
Re-track: git lfs track "t5-small/*"; git add .gitattributes t5-small/*.
Commit and push: git commit -m "Track t5-small with Git LFS"; git push origin main.




Authentication Errors:

Symptom: remote: Invalid username or password (HTTPS) or Permission denied (publickey) (SSH).
Fix:
HTTPS: Create PAT at GitHub Settings.
SSH: Generate key (ssh-keygen -t ed25519 -C "your_email@example.com"), add to GitHub, verify (ssh -T git@github.com).





Notes

Git LFS Storage: GitHub provides 1 GB free LFS storage. t5-small/ (~300 MB) and PDFs should fit. Check usage at GitHub Settings > Billing.
Offline Execution: No internet access required after Docker build.
Time Zone: Logs use UTC; local time is 12:07 AM IST, July 29, 2025.

