import os
import json
from datetime import datetime, UTC
import logging
import sys
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, pos_tag
from typing import Dict, List
import warnings
from synonyms import get_synonyms

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

import pdf_parser
from search_engine import SearchEngine
from result_generator import ImprovedResultGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

def load_config(path='config.json'):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        save_error_log({"error": "Failed to load config", "details": str(e)})
        raise

def create_dynamic_ideal_document(query: str, persona: str, config: Dict) -> str:
    tokens = word_tokenize(query.lower())
    tagged = pos_tag(tokens)
    
    key_terms = [word for word, pos in zip(tokens, [t[1] for t in tagged]) if pos.startswith(('NN', 'VB', 'JJ')) and len(word) > 3]
    entities = [word for word, pos in zip(tokens, [t[1] for t in tagged]) if pos.startswith('NN') and len(word) > 3]
    intent = next((word for word, pos in zip(tokens, [t[1] for t in tagged]) if pos.startswith('VB') and len(word) > 3), "address")
    
    synonyms = []
    for term in key_terms:
        synonyms.extend(get_synonyms(term)[:2])
    key_terms.extend(synonyms)
    
    activity_keywords = []
    for keywords in config.get('activity_keywords', {}).values():
        activity_keywords.extend(keywords)
    irrelevant_terms = config.get('irrelevant_terms', ['introduction', 'conclusion', 'summary', 'overview'])
    
    relevant_activities = [kw for kw in activity_keywords if any(term in kw.lower() for term in key_terms)]
    if not relevant_activities:
        relevant_activities = activity_keywords[:5] or ['relevant topics']
        
    persona_clean = persona.lower().replace('_', ' ')
    
    base = f"relevant and actionable information tailored to the needs of a {persona_clean}"
    focus = ', '.join(relevant_activities[:5] or key_terms[:5] or ['specific topics'])
    
    ideal_doc = f"""
    This document provides {base} to {intent} the query '{query}'. 
    It focuses on {focus} and includes specific, practical information about {', '.join(key_terms[:5] or ['the topic'])}.
    The content is detailed, actionable, and directly relevant to {', '.join(entities[:3] or ['the subject'])}.
    It emphasizes {', '.join(relevant_activities[:3] or ['relevant and specific content'])} and avoids {', '.join(irrelevant_terms[:3] or ['generic or irrelevant content'])}.
    It contains concrete examples, specific recommendations, and practical steps to achieve the goal.
    """
    
    return ideal_doc.strip()

def save_chunks(chunks: List[Dict], output_path: str = './output/chunks.json'):
    logger.info(f"Saving {len(chunks)} chunks to {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)

def save_error_log(error_data: Dict, output_path: str = './output/error_log.json'):
    logger.info(f"Saving error log to {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2)
        f.write('\n')

def main():
    setup_nltk()
    
    # Validate command-line arguments
    if len(sys.argv) != 3:
        error_msg = "Usage: python main.py <query> <persona>"
        logger.error(error_msg)
        save_error_log({"error": "Invalid command-line arguments", "details": error_msg})
        sys.exit(1)
    
    query, persona = sys.argv[1], sys.argv[2]
    if not query.strip() or not persona.strip():
        error_msg = "Query and persona must be non-empty strings"
        logger.error(error_msg)
        save_error_log({"error": "Empty query or persona", "details": error_msg})
        sys.exit(1)
    
    logger.info(f"Processing query: '{query}' for persona: '{persona}'")
    config = load_config()

    all_chunks = []
    for pdf_file in os.listdir(config['pdf_data_path']):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(config['pdf_data_path'], pdf_file)
            try:
                all_chunks.extend(pdf_parser.process_pdf_to_chunks(pdf_path, **config.get('pdf_parsing_parameters', {})))
            except Exception as e:
                logger.warning(f"Failed to process {pdf_file}: {e}")
                save_error_log({"file": pdf_file, "error": str(e)})

    if not all_chunks:
        logger.error("No chunks extracted from PDFs. Check your PDF files or parameters.")
        save_error_log({"error": "No chunks extracted"})
        return

    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    save_chunks(all_chunks)

    try:
        engine = SearchEngine(config['model_paths'])
        engine.build_index(all_chunks)
    except Exception as e:
        logger.error(f"Failed to build search index: {e}")
        save_error_log({"error": "Failed to build search index", "details": str(e)})
        return

    start_time = datetime.now(UTC)
    enhanced_query = f"As a {persona}, I need information about: {query}"
    ideal_document = create_dynamic_ideal_document(query, persona, config)
    try:
        ideal_doc_vector = engine.bi_encoder.encode([ideal_document])
    except Exception as e:
        logger.error(f"Failed to encode ideal document: {e}")
        save_error_log({"error": "Failed to encode ideal document", "details": str(e)})
        return

    top_k = config['search_parameters'].get('top_k_retrieval', 100)
    if len(query.split()) > 10:
        top_k = int(top_k * 1.5)

    try:
        candidate_items = engine.search_and_rerank(
            enhanced_query,
            top_k,
            config['search_parameters']['top_k_rerank']
        )
    except Exception as e:
        logger.error(f"Search and rerank failed: {e}")
        save_error_log({"error": "Search and rerank failed", "details": str(e)})
        return

    if not candidate_items:
        logger.warning("No candidate items found. Check search parameters or content.")
        save_error_log({"error": "No candidate items found"})
        return

    intent_weight = config['filtering_parameters'].get('intent_weight', 0.6)
    for item in candidate_items:
        chunk_embedding = engine.chunk_embeddings[item['original_index']].reshape(1, -1)
        intent_score = float(cosine_similarity(ideal_doc_vector, chunk_embedding)[0][0])
        item['blended_score'] = float((item['rerank_score'] * (1.0 - intent_weight)) + (intent_score * intent_weight))

    final_items = sorted(candidate_items, key=lambda x: x['blended_score'], reverse=True)

    logger.info(f"Re-ranked {len(final_items)} items using blended scoring")
    logger.info(f"Top score: {final_items[0]['blended_score']:.3f}")

    result_gen = ImprovedResultGenerator(
        config['model_paths']['summarizer'],
        bi_encoder=engine.bi_encoder,
        chunks=all_chunks,
        persona=persona
    )

    try:
        enhanced_results = result_gen.generate_enhanced_results(final_items, query)
    except Exception as e:
        logger.error(f"Failed to generate enhanced results: {e}")
        save_error_log({"error": "Failed to generate enhanced results", "details": str(e)})
        return

    final_output = {
        "metadata": {
            "input_documents": [f for f in os.listdir(config['pdf_data_path']) if f.endswith('.pdf')],
            "persona": persona,
            "job_to_be_done": query,
            "processing_timestamp": start_time.isoformat(),
            "enhanced_query": enhanced_query,
            "ideal_document_used": ideal_document,
            "processing_stats": enhanced_results["processing_stats"]
        },
        "extracted_sections": enhanced_results["sections"],
        "subsection_analysis": enhanced_results["subsections"]
    }

    output_dir = config['output_path']
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, 'result.json')

    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_serializable(final_output), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
        save_error_log({"error": "Failed to save result", "details": str(e)})
        return

    logger.info(f"Processing complete. Results saved to {result_file}")
    logger.info(f"Generated {len(enhanced_results['sections'])} sections and {len(enhanced_results['subsections'])} subsections")

    engine.release_models()

if __name__ == "__main__":
    main()