Document Relevance Processor

Overview

The Document Relevance Processor is a Python-based tool designed to rank and summarize document sections based on their relevance to a user query. It leverages natural language processing (NLP) techniques, including T5 for summarization, BM25 for ranking, and TF-IDF for term extraction, to filter out irrelevant content and prioritize sections matching the query. The tool supports persona-based filtering (e.g., "Health_Conscious") and is ideal for tasks like generating vegetarian menu suggestions from document sets. It processes document chunks, applies relevance scoring, and outputs structured results in JSON format.
Prerequisites

Docker: Ensure Docker is installed to run the application in a containerized environment.
Input Files: Provide config.json, irrelevant_terms.json, relevant_terms.json, and a synonyms.py module for synonym generation.
Document Chunks: Supply document chunks in a compatible format (e.g., JSON with text and metadata).

Setup Instructions

Clone the Repository:git clone <repository-url>
cd document-relevance-processor


Prepare Input Files:
Place config.json, irrelevant_terms.json, relevant_terms.json, and synonyms.py in the project directory.
Ensure synonyms.py implements a get_synonyms function compatible with NLTK WordNet.


Create requirements.txt:transformers==4.31.0
sentencepiece==0.1.99
torch==2.0.1
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
rank-bm25==0.2.2
sentence-transformers==2.2.2


Build the Docker Image:docker build -t doc-relevance-processor .


Run the Docker Container:docker run -v $(pwd):/app -it doc-relevance-processor



Usage

Run the Script:Execute main.py (or modify result_generator.py to accept command-line arguments) with a query and persona:python main.py "Prepare a vegetarian buffet-style dinner menu" "Health_Conscious"


Output:
result.json: Contains ranked sections, summarized subsections, and processing stats.
chunk_scores_output.txt: Logs chunk scores, penalties, and term lists.


Customization:
Edit irrelevant_terms.json and relevant_terms.json to adjust filtering terms.
Modify config.json to update activity keywords.



Notes

Ensure sufficient memory for NLP models (T5 and bi-encoder).
If synonyms.py is missing, implement a basic version using NLTK WordNet.
For issues, check logs in output/error_log.json.
