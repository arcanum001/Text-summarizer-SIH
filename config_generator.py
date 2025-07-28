import os
import json
import fitz
from collections import defaultdict
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

def _extract_keywords_from_texts(texts: List[str], bi_encoder: SentenceTransformer) -> Tuple[Dict[str, List[str]], List[str]]:
    """Extract and cluster keywords from PDF texts, including irrelevant terms"""
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    
    keywords = defaultdict(int)
    irrelevant_candidates = defaultdict(int)
    
    # Extract keywords from all texts
    for text in texts:
        try:
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            for i, (word, pos) in enumerate(tagged):
                # Focus on nouns and noun phrases for activities
                if pos.startswith('NN') and word not in stop_words and len(word) > 3:
                    keywords[word] += 1
                    if i < len(tagged) - 1 and tagged[i+1][1].startswith('NN'):
                        phrase = f"{word} {tagged[i+1][0]}"
                        keywords[phrase] += 1
                # Identify irrelevant terms (adjectives, generic terms)
                if pos in ('JJ', 'IN', 'DT') and word not in stop_words and len(word) > 3:
                    irrelevant_candidates[word] += 1
                # Penalize budget/family/generic terms
                if word in ['budget', 'affordable', 'cheap', 'inexpensive', 'family', 'kids', 'children', 
                           'introduction', 'conclusion', 'summary', 'overview', 'general', 'basic']:
                    irrelevant_candidates[word] += 5
        except Exception as e:
            logger.warning(f"Error processing text for keywords: {e}")
            continue
    
    # Get top keywords
    keyword_list = [k for k, v in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:50]]
    irrelevant_terms = [k for k, v in sorted(irrelevant_candidates.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    if not keyword_list:
        logger.warning("No keywords extracted from PDFs")
        return {"General Activities": ['activity', 'event', 'tour']}, ['introduction', 'conclusion', 'summary']
    
    try:
        # Cluster keywords using embeddings
        keyword_embeddings = bi_encoder.encode(keyword_list, batch_size=32)
        n_clusters = min(5, len(keyword_list))
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(keyword_embeddings)
        else:
            labels = [0] * len(keyword_list)
        
        # Group keywords by cluster
        activity_keywords = defaultdict(list)
        cluster_keywords = defaultdict(list)
        
        for keyword, label in zip(keyword_list, labels):
            cluster_keywords[label].append(keyword)
        
        # Create meaningful category names
        for label, clustered_kws in cluster_keywords.items():
            if not clustered_kws:
                continue
            representative_keyword = max(clustered_kws, key=lambda k: keywords[k])
            category_name = ' '.join(w.capitalize() for w in representative_keyword.split('_'))
            if len(category_name) < 3:
                category_name = f"Activity Category {label + 1}"
            activity_keywords[category_name] = list(set(clustered_kws))
        
        if not activity_keywords:
            activity_keywords["General Activities"] = keyword_list[:10]
        
        if not irrelevant_terms:
            irrelevant_terms = ['introduction', 'conclusion', 'summary', 'overview']
        
        logger.info(f"Extracted {len(activity_keywords)} keyword categories and {len(irrelevant_terms)} irrelevant terms")
        return dict(activity_keywords), irrelevant_terms
        
    except Exception as e:
        logger.error(f"Error clustering keywords: {e}")
        return {"General Activities": keyword_list[:10]}, ['introduction', 'conclusion', 'summary']

def extract_pdf_metadata(pdf_directory: str) -> Dict:
    """Extract metadata from PDFs to inform configuration"""
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
    
    metadata = {
        "total_pdfs": len(pdf_files),
        "pdf_names": [os.path.basename(f) for f in pdf_files],
        "total_pages": 0,
        "avg_text_per_page": 0,
        "document_types": []
    }
    
    total_text_length = 0
    
    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as doc:
                metadata["total_pages"] += len(doc)
                sample_pages = min(3, len(doc))
                for i in range(sample_pages):
                    page_text = doc[i].get_text("text", sort=True)
                    total_text_length += len(page_text)
        except Exception as e:
            logger.warning(f"Could not analyze {pdf_path}: {e}")
    
    if metadata["total_pages"] > 0:
        metadata["avg_text_per_page"] = total_text_length // metadata["total_pages"]
    
    return metadata

def generate_config_from_pdfs(pdf_directory: str, model_path: str, output_path: str = 'config.json') -> Dict:
    """Generate optimized configuration based on PDF content analysis"""
    logger.info("Dynamically generating configuration from PDF content...")
    
    pdf_metadata = extract_pdf_metadata(pdf_directory)
    logger.info(f"Analyzed {pdf_metadata['total_pdfs']} PDFs with {pdf_metadata['total_pages']} total pages")
    
    texts = []
    for pdf_path in [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]:
        try:
            with fitz.open(pdf_path) as doc:
                sample_pages = min(5, len(doc))
                for page_num in range(sample_pages):
                    page_text = doc[page_num].get_text("text", sort=True)
                    if page_text.strip():
                        texts.append(page_text[:1000])
        except Exception as e:
            logger.warning(f"Could not read {pdf_path}: {e}")
    
    if not texts:
        raise ValueError("No text could be extracted from PDFs.")
    
    try:
        bi_encoder = SentenceTransformer(model_path)
        activity_keywords, irrelevant_terms = _extract_keywords_from_texts(texts, bi_encoder)
    except Exception as e:
        logger.error(f"Could not load model or extract keywords: {e}")
        activity_keywords, irrelevant_terms = {"General Activities": ['activity', 'event', 'tour']}, ['introduction', 'conclusion', 'summary']
    
    config = {
        "model_paths": {
            "bi_encoder": "./models/bi_encoder",
            "cross_encoder": "./models/cross_encoder",
            "summarizer": "./models/summarizer"
        },
        "pdf_data_path": pdf_directory,
        "output_path": "./output",
        "search_parameters": {
            "top_k_retrieval": min(150, max(50, pdf_metadata["total_pages"] * 2)),
            "top_k_rerank": min(75, max(25, pdf_metadata["total_pages"]))
        },
        "pdf_parsing_parameters": {
            "min_chunk_word_count": 20 if pdf_metadata["avg_text_per_page"] > 500 else 15,
            "max_heading_word_count": 15
        },
        "filtering_parameters": {
            "intent_score_weight": 0.6
        },
        "pdf_metadata": pdf_metadata,
        "activity_keywords": activity_keywords,
        "irrelevant_terms": irrelevant_terms
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Successfully generated and saved configuration to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise
    
    return config

def validate_config(config_path: str) -> bool:
    """Validate that the configuration file is properly formatted"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['model_paths', 'pdf_data_path', 'search_parameters', 'pdf_parsing_parameters', 'activity_keywords', 'irrelevant_terms']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        for model_name, model_path in config['model_paths'].items():
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
        
        logger.info("Configuration validation successful")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python config_generator.py <pdf_directory> <model_path> [output_path]")
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'config.json'
    
    try:
        config = generate_config_from_pdfs(pdf_dir, model_path, output_path)
        print(f"Configuration generated successfully: {output_path}")
        
        if validate_config(output_path):
            print("Configuration validation passed!")
        else:
            print("Configuration validation failed!")
    except Exception as e:
        print(f"Error generating configuration: {e}")
        sys.exit(1)