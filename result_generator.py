from collections import defaultdict
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import logging
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from synonyms import get_synonyms

logger = logging.getLogger(__name__)

def generate_irrelevant_and_relevant_terms(query: str, chunks: List[Dict], bi_encoder=None, persona: str = "") -> tuple[List[str], List[str]]:
    try:
        # Load datasets
        try:
            with open('irrelevant_terms.json', 'r') as f:
                irrelevant_dataset = json.load(f)
            with open('relevant_terms.json', 'r') as f:
                relevant_dataset = json.load(f)
            logger.debug(f"Loaded datasets: irrelevant={list(irrelevant_dataset.keys())[:10]}, relevant={list(relevant_dataset.keys())[:10]}")
        except Exception as e:
            logger.warning(f"Failed to load datasets: {e}")
            irrelevant_dataset = {}
            relevant_dataset = {}

        # Default terms
        default_irrelevant = [
            'introduction', 'conclusion', 'summary', 'overview', 'instructions',
            'preface', 'foreword', 'abstract', 'prologue', 'epilogue',
            'contents', 'index', 'appendix', 'glossary', 'bibliography',
            'references', 'acknowledgments', 'notes', 'disclaimer', 'guidelines',
            'procedure', 'methodology', 'background', 'table of contents', 'cover page'
        ]
        generic_synonyms = []
        for term in default_irrelevant:
            generic_synonyms.extend([syn for syn in get_synonyms(term)[:2] if len(syn.split()) <= 2 and len(syn) > 2])
        default_irrelevant.extend(generic_synonyms)

        # Extract query keywords
        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        query_keywords = [word for word, pos in zip(tokens, [t[1] for t in tagged]) if pos.startswith(('NN', 'JJ', 'VB')) and len(word) > 3]
        logger.debug(f"Query keywords extracted: {query_keywords}")

        # Initialize term sets
        irrelevant_terms = set(default_irrelevant)
        relevant_terms = set()

        # Persona matching
        selected_persona = None
        selected_domain = None
        if irrelevant_dataset and relevant_dataset:
            for domain, personas in irrelevant_dataset.get('domains', {}).items():
                for p, data in personas.items():
                    query_keywords_list = data.get('query_keywords', [])
                    # Match by persona (case-insensitive) or query keywords
                    if p.lower() == persona.lower() or any(kw.lower() in query.lower() for kw in query_keywords_list):
                        selected_persona = p
                        selected_domain = domain
                        irrelevant_terms.update([term.lower() for term in data.get('irrelevant_terms', []) if len(term) > 2])
                        relevant_terms.update(
                            [term.lower() for term in relevant_dataset.get('domains', {})
                            .get(domain, {})
                            .get('personas', {})
                            .get(p, {})
                            .get('relevant_terms', []) if len(term) > 2]
                        )
                        logger.debug(f"Matched persona: {selected_persona}, domain: {selected_domain}")
                        logger.debug(f"Dataset irrelevant terms: {list(irrelevant_terms)[:20]}")
                        logger.debug(f"Dataset relevant terms: {list(relevant_terms)[:20]}")
                        break
                if selected_persona:
                    break

        # Fallback to existing method if no persona match or datasets are invalid
        if not selected_persona:
            logger.info("No persona match found, using fallback method")
            for term in query_keywords:
                for synset in wordnet.synsets(term, pos=[wordnet.NOUN, wordnet.ADJ, wordnet.VERB]):
                    for lemma in synset.lemmas():
                        for antonym in lemma.antonyms():
                            ant_term = antonym.name().replace('_', ' ').lower()
                            if len(ant_term.split()) <= 2 and len(ant_term) > 2:
                                irrelevant_terms.add(ant_term)
                    for hyponym in synset.hyponyms():
                        for lemma in hyponym.lemmas():
                            hyp_term = lemma.name().replace('_', ' ').lower()
                            if len(hyp_term.split()) <= 2 and len(hyp_term) > 2:
                                irrelevant_terms.add(hyp_term)

            texts = [chunk['text'] for chunk in chunks if chunk.get('text')]
            if texts:
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.6)
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                term_scores = tfidf_matrix.mean(axis=0).A1
                top_terms = [feature_names[i] for i in term_scores.argsort()[-30:][::-1]]
                query_related = query_keywords + [syn for term in query_keywords for syn in get_synonyms(term)[:3]]
                irrelevant_terms.update(t for t in top_terms if len(t) > 2 and not any(q in t.lower() for q in query_related))
                relevant_terms.update(t for t in top_terms if len(t) > 2 and any(q in t.lower() for q in query_related))
                logger.debug(f"Fallback irrelevant terms: {list(irrelevant_terms)[:20]}")
                logger.debug(f"Fallback relevant terms: {list(relevant_terms)[:20]}")

            if bi_encoder and texts:
                query_embedding = bi_encoder.encode([query])[0]
                term_embeddings = bi_encoder.encode(list(irrelevant_terms))
                similarities = cosine_similarity([query_embedding], term_embeddings)[0]
                irrelevant_terms = {term for term, sim in zip(irrelevant_terms, similarities) if sim < 0.15}

        # Ensure no overlap and filter
        irrelevant_terms = [t for t in irrelevant_terms if t not in query_keywords and t not in relevant_terms]
        relevant_terms = [t for t in relevant_terms if any(q in t.lower() for q in query_keywords) or t in relevant_dataset.get('domains', {}).get(selected_domain, {}).get('personas', {}).get(selected_persona, {}).get('relevant_terms', [])]
        irrelevant_terms = sorted(list(irrelevant_terms), key=len, reverse=True)[:50]
        relevant_terms = sorted(list(relevant_terms), key=len, reverse=True)[:50]

        logger.debug(f"Final irrelevant terms: {irrelevant_terms[:20]}")
        logger.debug(f"Final relevant terms: {relevant_terms[:20]}")
        return irrelevant_terms, relevant_terms
    except Exception as e:
        logger.warning(f"Failed to generate terms: {e}")
        return default_irrelevant, []

class ImprovedResultGenerator:
    def __init__(self, model_path: str, bi_encoder=None, chunks: List[Dict] = None, persona: str = ""):
        tokenizer = BartTokenizer.from_pretrained(model_path, local_files_only=True)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device='cpu')
        self.bi_encoder = bi_encoder
        self.chunks = chunks or []
        self.persona = persona
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            self.irrelevant_terms, self.relevant_terms = generate_irrelevant_and_relevant_terms("", self.chunks, self.bi_encoder, self.persona)
            self.activity_keywords = config.get('activity_keywords', {
                "General Activities": ['activity', 'event', 'recipe', 'menu', 'itinerary', 'dish', 'course', 'buffet', 'dinner', 'salad', 'curry'],
                "PDF Forms": ['form', 'fillable', 'e-signature', 'signature', 'document', 'field'],
                "Business": ['contract', 'agreement', 'proposal', 'compliance', 'onboarding']
            })
            self.query_keywords = []
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            self.irrelevant_terms, self.relevant_terms = generate_irrelevant_and_relevant_terms("", self.chunks, self.bi_encoder, self.persona)
            self.activity_keywords = {
                "General Activities": ['activity', 'event', 'recipe', 'menu', 'itinerary', 'dish', 'course', 'buffet', 'dinner', 'salad', 'curry'],
                "PDF Forms": ['form', 'fillable', 'e-signature', 'signature', 'document', 'field'],
                "Business": ['contract', 'agreement', 'proposal', 'compliance', 'onboarding']
            }
            self.query_keywords = []

    def update_query_context(self, query: str):
        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        self.query_keywords = [word for word, pos in zip(tokens, [t[1] for t in tagged]) 
                             if pos.startswith(('NN', 'JJ', 'VB')) and len(word) > 3]
        self.query_keywords = list(set(self.query_keywords + [syn for term in self.query_keywords 
                                                            for syn in get_synonyms(term)[:3] 
                                                            if len(syn.split()) <= 2 and len(syn) > 2]))
        texts = [chunk['text'] for chunk in self.chunks if chunk.get('text')]
        if texts:
            vectorizer = TfidfVectorizer(max_features=30, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            term_scores = tfidf_matrix.mean(axis=0).A1
            top_terms = [feature_names[i] for i in term_scores.argsort()[-15:][::-1]]
            self.query_keywords.extend([t for t in top_terms if any(q in t.lower() for q in self.query_keywords)])
        if self.bi_encoder and texts:
            query_embedding = self.bi_encoder.encode([query])[0]
            chunk_embeddings = self.bi_encoder.encode(texts)
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            top_indices = similarities.argsort()[-10:][::-1]
            for idx in top_indices:
                chunk_text = texts[idx].lower().split()
                self.query_keywords.extend([word for word in chunk_text if word not in self.query_keywords and len(word) > 3][:3])
        self.query_keywords = list(set(self.query_keywords))
        self.irrelevant_terms, self.relevant_terms = generate_irrelevant_and_relevant_terms(query, self.chunks, self.bi_encoder, self.persona)
        logger.debug(f"Updated query keywords: {self.query_keywords[:20]}")

    def _calculate_section_relevance_score(self, section_chunks: List[Dict], query: str) -> tuple[float, float, str]:
        if not section_chunks:
            return 0.0, 0.0, "No chunks provided"

        texts = [chunk['chunk']['text'] for chunk in section_chunks if chunk.get('chunk') and chunk['chunk'].get('text')]
        section_title = section_chunks[0]['chunk']['metadata']['section_title'].lower()
        content_density = np.mean([chunk['chunk']['metadata'].get('content_density', 0.5) for chunk in section_chunks])

        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        query_tokens = query.lower().split() + self.query_keywords
        bm25_scores = bm25.get_scores(query_tokens)
        avg_bm25 = np.mean(bm25_scores) / 10.0 if bm25_scores.size else 0.0

        avg_score = np.mean([chunk['blended_score'] for chunk in section_chunks])
        avg_score = max(0.0, min(1.0, avg_score))

        high_quality_chunks = sum(1 for chunk in section_chunks if chunk['blended_score'] > 0.6)
        density_score = min(high_quality_chunks / 10.0 + content_density * 0.3, 1.0)

        if len(section_chunks) > 1 and self.bi_encoder:
            embeddings = self.bi_encoder.encode(texts)
            similarities = cosine_similarity(embeddings)
            coherence_score = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        else:
            coherence_score = 0.5

        if self.bi_encoder:
            title_embedding = self.bi_encoder.encode([section_title])[0]
            query_embedding = self.bi_encoder.encode([query])[0]
            title_relevance = cosine_similarity([title_embedding], [query_embedding])[0][0]
            activity_score = 0.0
            for keywords in self.activity_keywords.values():
                if any(kw in section_title or kw in ' '.join(texts).lower() for kw in keywords):
                    activity_score = max(activity_score, 0.5)
            query_term_score = 0.0
            if any(kw in section_title or kw in ' '.join(texts).lower() for kw in self.query_keywords):
                query_term_score = 0.8
            title_relevance = min(title_relevance + activity_score + query_term_score, 1.0)
        else:
            title_relevance = 0.5
            activity_score = 0.0
            for keywords in self.activity_keywords.values():
                if any(kw in section_title or kw in ' '.join(texts).lower() for kw in keywords):
                    activity_score = max(activity_score, 0.5)
            query_term_score = 0.0
            if any(kw in section_title or kw in ' '.join(texts).lower() for kw in self.query_keywords):
                query_term_score = 0.8
            title_relevance = min(title_relevance + activity_score + query_term_score, 1.0)

        penalty = 0.0
        penalty_reason = "None"
        boost = 0.0
        combined_text = section_title + ' ' + ' '.join(texts).lower()
        matched_irrelevant = [term for term in self.irrelevant_terms if term in combined_text]
        matched_relevant = [term for term in self.relevant_terms if term in combined_text]
        if matched_irrelevant:
            penalty = 0.98
            penalty_reason = f"Contains irrelevant terms: {', '.join(matched_irrelevant[:5])}"
            logger.info(f"Applied penalty to {section_title}: {matched_irrelevant[:5]}")
        if matched_relevant:
            boost = 0.3
            logger.info(f"Applied boost to {section_title}: {matched_relevant[:5]}")

        structural_boost = 0.3 if any(chunk['chunk']['metadata'].get('is_list_or_table', False) for chunk in section_chunks) else 0.0
        hierarchy_boost = 0.2 / max(1, min(chunk['chunk']['metadata'].get('hierarchy_level', 1) for chunk in section_chunks))

        final_score = (
            avg_score * 0.15 +
            density_score * 0.1 +
            coherence_score * 0.1 +
            title_relevance * 0.4 +
            avg_bm25 * 0.35 +
            structural_boost +
            hierarchy_boost +
            boost
        ) * (1.0 - penalty)

        logger.debug(f"Section {section_title} score: {final_score:.3f} (bm25: {avg_bm25:.3f}, title: {title_relevance:.3f}, penalty: {penalty:.3f}, boost: {boost:.3f})")
        return max(0.0, min(1.0, float(final_score))), penalty, penalty_reason

    def generate_sections(self, ranked_items: List[Dict], query: str = "") -> Dict:
        if not ranked_items:
            return {"sections": [], "chunk_scores": [], "irrelevant_terms": [], "relevant_terms": []}

        self.update_query_context(query)

        section_groups = defaultdict(list)
        chunk_scores = []
        for item in ranked_items:
            try:
                section_title = item['chunk']['metadata']['section_title'].lower()
                generic_keywords = ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']
                text = item['chunk']['text'].lower() if item.get('chunk') and item['chunk'].get('text') else ''
                if any(keyword in section_title for keyword in generic_keywords):
                    logger.info(f"Excluded section due to generic title: {section_title}")
                    chunk_scores.append({
                        "document": item['chunk']['metadata']['doc_name'],
                        "page_number": int(item['chunk']['metadata']['page_number']),
                        "section_title": section_title,
                        "chunk_text": text[:100] + "..." if len(text) > 100 else text,
                        "blended_score": float(round(item.get('blended_score', 0.5), 3)),
                        "bm25_score": 0.0,
                        "relevance_score": 0.0,
                        "status": "Excluded",
                        "penalty": 0.0,
                        "penalty_reason": f"Generic title '{section_title}'",
                        "boost": 0.0
                    })
                    continue
                section_groups[section_title].append(item)

                chunk_text = [text]
                tokenized_text = [text.lower().split()]
                bm25 = BM25Okapi(tokenized_text)
                query_tokens = query.lower().split() + self.query_keywords
                bm25_scores = bm25.get_scores(query_tokens)
                bm25_score = bm25_scores[0] / 10.0 if bm25_scores.size else 0.0
                blended_score = item.get('blended_score', 0.5)
                content_density = item['chunk']['metadata'].get('content_density', 0.5)
                density_score = content_density * 0.3
                coherence_score = 0.5
                title_relevance = 0.5
                if self.bi_encoder:
                    title_embedding = self.bi_encoder.encode([section_title])[0]
                    query_embedding = self.bi_encoder.encode([query])[0]
                    title_relevance = cosine_similarity([title_embedding], [query_embedding])[0][0]
                    if any(kw in section_title or kw in text for kw in self.activity_keywords['General Activities']):
                        title_relevance += 0.5
                    if any(kw in section_title or kw in text for kw in self.query_keywords):
                        title_relevance += 0.8
                    title_relevance = min(title_relevance, 1.0)
                penalty = 0.0
                penalty_reason = "None"
                boost = 0.0
                matched_irrelevant = [term for term in self.irrelevant_terms if term in (section_title + ' ' + text)]
                matched_relevant = [term for term in self.relevant_terms if term in (section_title + ' ' + text)]
                if matched_irrelevant:
                    penalty = 0.98
                    penalty_reason = f"Contains irrelevant terms: {', '.join(matched_irrelevant[:5])}"
                if matched_relevant:
                    boost = 0.3
                chunk_relevance_score = (
                    blended_score * 0.15 +
                    density_score * 0.1 +
                    coherence_score * 0.1 +
                    title_relevance * 0.4 +
                    bm25_score * 0.35 +
                    boost
                ) * (1.0 - penalty)
                chunk_scores.append({
                    "document": item['chunk']['metadata']['doc_name'],
                    "page_number": int(item['chunk']['metadata']['page_number']),
                    "section_title": section_title,
                    "chunk_text": text[:100] + "..." if len(text) > 100 else text,
                    "blended_score": float(round(blended_score, 3)),
                    "bm25_score": float(round(bm25_score, 3)),
                    "relevance_score": float(round(chunk_relevance_score, 3)),
                    "status": "Excluded" if penalty > 0 or chunk_relevance_score < 0.6 else "Included",
                    "penalty": float(round(penalty, 3)),
                    "penalty_reason": penalty_reason,
                    "boost": float(round(boost, 3))
                })
                logger.debug(f"Chunk {section_title} (page {item['chunk']['metadata']['page_number']}): relevance_score={chunk_relevance_score:.3f}, blended_score={blended_score:.3f}, bm25_score={bm25_score:.3f}, penalty={penalty:.3f}, boost={boost:.3f}, reason={penalty_reason}")
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}, {json.dumps(item, indent=2)}")
                chunk_scores.append({
                    "document": item['chunk']['metadata'].get('doc_name', 'unknown'),
                    "page_number": int(item['chunk']['metadata'].get('page_number', 0)),
                    "section_title": section_title,
                    "chunk_text": "Error processing chunk",
                    "blended_score": float(round(item.get('blended_score', 0.5), 3)),
                    "bm25_score": 0.0,
                    "relevance_score": 0.0,
                    "status": "Excluded",
                    "penalty": 0.0,
                    "penalty_reason": f"Error processing: {str(e)}",
                    "boost": 0.0
                })
                continue

        section_scores = []
        for section_title, chunks in section_groups.items():
            try:
                relevance_score, penalty, penalty_reason = self._calculate_section_relevance_score(chunks, query)
                if relevance_score < 0.6:
                    logger.info(f"Excluded low-relevance section: {section_title} (score: {relevance_score:.3f})")
                    continue
                combined_text = section_title + ' ' + ' '.join(chunk['chunk']['text'].lower() for chunk in chunks if chunk.get('chunk') and chunk['chunk'].get('text'))
                if any(term in combined_text for term in self.irrelevant_terms):
                    logger.info(f"Excluded section with irrelevant terms: {section_title} (terms: {self.irrelevant_terms[:10]})")
                    continue
                best_chunk = max(chunks, key=lambda x: x.get('blended_score', 0))
                section_scores.append({
                    'section_title': section_title,
                    'relevance_score': relevance_score,
                    'chunk_count': len(chunks),
                    'best_chunk': best_chunk,
                    'chunks': chunks
                })
            except Exception as e:
                logger.warning(f"Error processing section {section_title}: {e}")
                continue

        section_scores.sort(key=lambda x: x['relevance_score'], reverse=True)

        sections = []
        used_titles = set()
        used_docs = set()
        for i, section_data in enumerate(section_scores):
            if len(sections) >= 5:
                break
            if section_data['section_title'] in used_titles:
                continue
            best_chunk = section_data['best_chunk']
            doc_name = best_chunk['chunk']['metadata']['doc_name']
            if doc_name in used_docs and len(sections) > 2:
                continue
            combined_text = section_data['section_title'] + ' ' + ' '.join(chunk['chunk']['text'].lower() for chunk in section_data['chunks'] if chunk.get('chunk') and chunk['chunk'].get('text'))
            if any(term in combined_text for term in self.irrelevant_terms):
                logger.info(f"Excluded section during final check: {section_data['section_title']} (terms: {self.irrelevant_terms[:10]})")
                continue
            sections.append({
                "document": doc_name,
                "page_number": int(best_chunk['chunk']['metadata']['page_number']),
                "section_title": section_data['section_title'],
                "importance_rank": i + 1,
                "relevance_score": float(round(section_data['relevance_score'], 3)),
                "chunk_count": int(section_data['chunk_count'])
            })
            used_titles.add(section_data['section_title'])
            used_docs.add(doc_name)

        if len(sections) < 5:
            remaining = [item for item in ranked_items if item['chunk']['metadata']['section_title'].lower() not in used_titles]
            for item in sorted(remaining, key=lambda x: x.get('blended_score', 0), reverse=True):
                if len(sections) >= 5:
                    break
                section_title = item['chunk']['metadata']['section_title'].lower()
                doc_name = item['chunk']['metadata']['doc_name']
                if doc_name not in used_docs or len(sections) <= 2:
                    text = item['chunk']['text'].lower() if item.get('chunk') and item['chunk'].get('text') else ''
                    if any(term in text for term in self.irrelevant_terms):
                        logger.info(f"Excluded fallback section with irrelevant terms: {section_title} (terms: {self.irrelevant_terms[:10]})")
                        continue
                    if any(keyword in section_title for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                        logger.info(f"Excluded fallback section due to generic title: {section_title}")
                        continue
                    sections.append({
                        "document": doc_name,
                        "page_number": int(item['chunk']['metadata']['page_number']),
                        "section_title": section_title,
                        "importance_rank": len(sections) + 1,
                        "relevance_score": float(round(item.get('blended_score', 0), 3)),
                        "chunk_count": 1
                    })
                    used_titles.add(section_title)
                    used_docs.add(doc_name)

        # Write output to file
        with open("chunk_scores_output.txt", "w", encoding="utf-8") as f:
            f.write("=== Chunk Relevance Scores ===\n\n")
            for chunk in chunk_scores:
                f.write(f"Document: {chunk['document']}\n")
                f.write(f"Page Number: {chunk['page_number']}\n")
                f.write(f"Section Title: {chunk['section_title']}\n")
                f.write(f"Chunk Text: {chunk['chunk_text']}\n")
                f.write(f"Blended Score: {chunk['blended_score']:.3f}\n")
                f.write(f"BM25 Score: {chunk['bm25_score']:.3f}\n")
                f.write(f"Relevance Score: {chunk['relevance_score']:.3f}\n")
                f.write(f"Status: {chunk['status']}\n")
                f.write(f"Penalty: {chunk['penalty']:.3f}\n")
                f.write(f"Penalty Reason: {chunk['penalty_reason']}\n")
                f.write(f"Boost: {chunk['boost']:.3f}\n")
                f.write("-" * 50 + "\n\n")
            f.write("=== Irrelevant Terms ===\n\n")
            f.write("Dynamically Generated Irrelevant Terms:\n")
            f.write(", ".join(self.irrelevant_terms[:50]) + "\n")
            f.write("=== Relevant Terms ===\n\n")
            f.write("Dynamically Generated Relevant Terms:\n")
            f.write(", ".join(self.relevant_terms[:50]) + "\n")

        logger.debug(f"Generated {len(sections)} sections: {[s['section_title'] for s in sections]}")
        return {
            "sections": sections,
            "chunk_scores": chunk_scores,
            "irrelevant_terms": self.irrelevant_terms[:50],
            "relevant_terms": self.relevant_terms[:50]
        }

    def generate_subsections(self, ranked_items: List[Dict], chunk_embeddings: np.ndarray, query: str) -> List[Dict]:
        if not ranked_items:
            return []

        self.update_query_context(query)

        top_items = []
        for item in ranked_items[:50]:
            text = item['chunk']['text'].lower() if item.get('chunk') and item['chunk'].get('text') else ''
            title = item['chunk']['metadata']['section_title'].lower()
            if any(term in title or term in text for term in self.irrelevant_terms):
                logger.info(f"Excluded due to irrelevant term in {title}: {self.irrelevant_terms[:10]}")
                continue
            if any(keyword in title for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                logger.info(f"Excluded due to generic title: {title}")
                continue
            doc_name = item['chunk']['metadata']['doc_name'].lower()
            if any(keyword in doc_name for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                logger.info(f"Excluded due to blocked PDF title: {doc_name}")
                continue
            score_boost = 0.0
            if any(kw in text or kw in title for kw in self.query_keywords):
                score_boost += 0.8
            if any(kw in text or kw in title for kw in self.relevant_terms):
                score_boost += 0.3
            for keywords in self.activity_keywords.values():
                if any(kw in text or kw in title for kw in keywords):
                    score_boost += 0.5
            if self.bi_encoder:
                text_embedding = self.bi_encoder.encode([text])[0]
                query_embedding = self.bi_encoder.encode([query])[0]
                sim_score = cosine_similarity([text_embedding], [query_embedding])[0][0]
                score_boost += sim_score * 0.6
            item['boosted_score'] = item.get('blended_score', 0) + item.get('bm25_score', 0) + score_boost
            top_items.append(item)

        top_items.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)
        top_items = top_items[:50]

        page_groups = defaultdict(list)
        for item in top_items:
            page_num = item['chunk']['metadata']['page_number']
            page_groups[page_num].append(item)

        page_scores = []
        for page_num, chunks in page_groups.items():
            try:
                avg_score = np.mean([chunk.get('boosted_score', 0) for chunk in chunks])
            except Exception as e:
                logger.warning(f"Failed to compute mean score for page {page_num}: {e}")
                avg_score = 0.0
            activity_score = 0.0
            text_content = ' '.join(chunk['chunk']['text'].lower() for chunk in chunks if chunk.get('chunk') and chunk['chunk'].get('text'))
            for keywords in self.activity_keywords.values():
                if any(kw in text_content for kw in keywords):
                    activity_score += 0.5
            if any(kw in text_content for kw in self.query_keywords):
                activity_score += 0.8
            if any(kw in text_content for kw in self.relevant_terms):
                activity_score += 0.3
            if self.bi_encoder:
                text_embedding = self.bi_encoder.encode([text_content])[0]
                query_embedding = self.bi_encoder.encode([query])[0]
                sim_score = cosine_similarity([text_embedding], [query_embedding])[0][0]
                activity_score += sim_score * 0.6
            if any(term in text_content for term in self.irrelevant_terms):
                logger.info(f"Excluded page {page_num} due to irrelevant terms: {self.irrelevant_terms[:10]}")
                continue
            page_scores.append({
                'page_num': page_num,
                'avg_score': avg_score + activity_score,
                'chunks': chunks,
                'document': chunks[0]['chunk']['metadata']['doc_name']
            })

        page_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        selected_pages = page_scores[:10]

        subsections = []
        used_pages = set()
        used_docs = set()
        for i, page_data in enumerate(selected_pages):
            if len(subsections) >= 5:
                break
            page_num = page_data['page_num']
            if page_num in used_pages:
                continue
            doc_name = page_data['document']
            if doc_name in used_docs and len(subsections) > 2:
                continue
            chunks = sorted(page_data['chunks'], key=lambda x: x.get('boosted_score', 0), reverse=True)
            for chunk in chunks:
                if chunk['chunk']['metadata'].get('is_list_or_table', False):
                    chunk['boosted_score'] += 0.3
                chunk['boosted_score'] += 0.1 / max(1, chunk['chunk']['metadata'].get('hierarchy_level', 1))
            refined_chunks = [chunk for chunk in chunks if not any(term in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for term in self.irrelevant_terms)]
            if not refined_chunks:
                continue
            refined_text = "\n\n".join(chunk['chunk']['text'].strip() for chunk in refined_chunks if chunk.get('chunk') and chunk['chunk'].get('text') and chunk['chunk']['text'].strip())
            if any(term in refined_text.lower() for term in self.irrelevant_terms) or any(keyword in doc_name.lower() or keyword in refined_text.lower() for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                logger.info(f"Excluded subsection with irrelevant terms: {doc_name}, page {page_num} (terms: {self.irrelevant_terms[:10]})")
                continue
            subsections.append({
                "document": doc_name,
                "refined_text": refined_text,
                "page_number": int(page_num)
            })
            used_pages.add(page_num)
            used_docs.add(doc_name)

        if len(subsections) < 5:
            remaining_pages = [p for p in page_scores if p['page_num'] not in used_pages]
            remaining_pages.sort(key=lambda x: x['avg_score'], reverse=True)
            for page_data in remaining_pages:
                if len(subsections) >= 5:
                    break
                page_num = page_data['page_num']
                if page_num in used_pages:
                    continue
                doc_name = page_data['document']
                if doc_name in used_docs and len(subsections) > 2:
                    continue
                chunks = sorted(page_data['chunks'], key=lambda x: x.get('boosted_score', 0), reverse=True)
                for chunk in chunks:
                    if chunk['chunk']['metadata'].get('is_list_or_table', False):
                        chunk['boosted_score'] += 0.3
                    chunk['boosted_score'] += 0.1 / max(1, chunk['chunk']['metadata'].get('hierarchy_level', 1))
                refined_chunks = [chunk for chunk in chunks if not any(term in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for term in self.irrelevant_terms)]
                if not refined_chunks:
                    continue
                refined_text = "\n\n".join(chunk['chunk']['text'].strip() for chunk in refined_chunks if chunk.get('chunk') and chunk['chunk'].get('text') and chunk['chunk']['text'].strip())
                if any(term in refined_text.lower() for term in self.irrelevant_terms) or any(keyword in doc_name.lower() or keyword in refined_text.lower() for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                    logger.info(f"Excluded subsection with irrelevant terms: {doc_name}, page {page_num} (terms: {self.irrelevant_terms[:10]})")
                    continue
                subsections.append({
                    "document": doc_name,
                    "refined_text": refined_text,
                    "page_number": int(page_num)
                })
                used_pages.add(page_num)
                used_docs.add(doc_name)

        if len(subsections) < 5:
            for item in ranked_items:
                if len(subsections) >= 5:
                    break
                page_num = item['chunk']['metadata']['page_number']
                if page_num in used_pages:
                    continue
                doc_name = item['chunk']['metadata']['doc_name']
                if doc_name in used_docs and len(subsections) > 2:
                    continue
                page_chunks = [it for it in ranked_items if it['chunk']['metadata']['page_number'] == page_num]
                if not page_chunks:
                    continue
                for chunk in page_chunks:
                    chunk['boosted_score'] = chunk.get('blended_score', 0) + chunk.get('bm25_score', 0)
                    if chunk['chunk']['metadata'].get('is_list_or_table', False):
                        chunk['boosted_score'] += 0.3
                    if any(kw in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for kw in self.query_keywords):
                        chunk['boosted_score'] += 0.8
                    if any(kw in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for kw in self.relevant_terms):
                        chunk['boosted_score'] += 0.3
                    for keywords in self.activity_keywords.values():
                        if any(kw in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for kw in keywords):
                            chunk['boosted_score'] += 0.5
                    if self.bi_encoder:
                        text = chunk['chunk']['text'] if chunk.get('chunk') and chunk['chunk'].get('text') else ''
                        text_embedding = self.bi_encoder.encode([text])[0]
                        query_embedding = self.bi_encoder.encode([query])[0]
                        sim_score = cosine_similarity([text_embedding], [query_embedding])[0][0]
                        chunk['boosted_score'] += sim_score * 0.6
                refined_chunks = [chunk for chunk in page_chunks if not any(term in (chunk['chunk']['text'].lower() if chunk.get('chunk') and chunk['chunk'].get('text') else '') for term in self.irrelevant_terms)]
                if not refined_chunks:
                    continue
                refined_text = "\n\n".join(chunk['chunk']['text'].strip() for chunk in refined_chunks if chunk.get('chunk') and chunk['chunk'].get('text') and chunk['chunk']['text'].strip())
                if any(term in refined_text.lower() for term in self.irrelevant_terms) or any(keyword in doc_name.lower() or keyword in refined_text.lower() for keyword in ['introduction', 'conclusion', 'summary', 'overview', 'instructions', 'table of contents', 'cover page']):
                    logger.info(f"Excluded subsection with irrelevant terms: {doc_name}, page {page_num} (terms: {self.irrelevant_terms[:10]})")
                    continue
                subsections.append({
                    "document": doc_name,
                    "refined_text": refined_text,
                    "page_number": int(page_num)
                })
                used_pages.add(page_num)
                used_docs.add(doc_name)

        logger.debug(f"Generated {len(subsections)} subsections")
        return subsections[:5]

    def _extract_topics_from_chunks(self, chunks: List[Dict], n_topics: int = 5, query: str = "") -> List[Dict]:
        if len(chunks) < 2:
            return [{'chunks': chunks, 'topic_keywords': [], 'coherence_score': 1.0}]

        texts = [chunk['chunk']['text'] for chunk in chunks if chunk.get('chunk') and chunk['chunk'].get('text')]
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        query_tokens = query.lower().split() + self.query_keywords
        bm25_scores = bm25.get_scores(query_tokens)
        for i, chunk in enumerate(chunks):
            chunk['bm25_score'] = bm25_scores[i] if i < len(bm25_scores) else 0.0

        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.6
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            return [{'chunks': chunks, 'topic_keywords': [], 'coherence_score': 0.5}]

        n_clusters = min(n_topics, len(chunks))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
        else:
            cluster_labels = [0] * len(chunks)

        topics = []
        for cluster_id in range(n_clusters):
            cluster_chunks = [chunks[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            if not cluster_chunks:
                continue

            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            top_term_indices = cluster_tfidf.argsort()[-10:][::-1]
            topic_keywords = [feature_names[i] for i in top_term_indices if cluster_tfidf[i] > 0]

            filtered_keywords = []
            for kw in topic_keywords:
                if not any(term in kw.lower() for term in self.irrelevant_terms):
                    score = 1.0
                    for category, keywords in self.activity_keywords.items():
                        if any(k in kw.lower() for k in keywords):
                            score += 1.0
                    if any(k in kw.lower() for k in self.query_keywords):
                        score += 1.5
                    if any(k in kw.lower() for k in self.relevant_terms):
                        score += 0.5
                    filtered_keywords.append((kw, score))
            filtered_keywords.sort(key=lambda x: x[1], reverse=True)
            topic_keywords = [kw for kw, score in filtered_keywords[:5]]

            if len(cluster_chunks) > 1 and self.bi_encoder:
                cluster_texts = [chunk['chunk']['text'] for chunk in cluster_chunks if chunk.get('chunk') and chunk['chunk'].get('text')]
                embeddings = self.bi_encoder.encode(cluster_texts)
                similarities = cosine_similarity(embeddings)
                coherence = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            else:
                coherence = 1.0 if len(cluster_chunks) == 1 else 0.5

            avg_bm25 = np.mean([chunk['bm25_score'] for chunk in cluster_chunks])
            topics.append({
                'chunks': sorted(cluster_chunks, key=lambda x: x.get('blended_score', 0) + x.get('bm25_score', 0), reverse=True),
                'topic_keywords': topic_keywords,
                'coherence_score': float(coherence),
                'bm25_score': float(avg_bm25)
            })

        for topic in topics:
            avg_score = np.mean([chunk.get('blended_score', 0) + chunk.get('bm25_score', 0) for chunk in topic['chunks']])
            topic['quality_score'] = topic['coherence_score'] * avg_score

        topics.sort(key=lambda x: x['quality_score'], reverse=True)
        return topics[:5]

    def generate_enhanced_results(self, ranked_items: List[Dict], chunk_embeddings: np.ndarray, query: str = "") -> Dict:
        sections_data = self.generate_sections(ranked_items, query)
        subsections = self.generate_subsections(ranked_items, chunk_embeddings, query)

        return {
            "sections": sections_data["sections"],
            "subsections": subsections,
            "chunk_scores": sections_data["chunk_scores"],
            "irrelevant_terms": sections_data["irrelevant_terms"],
            "relevant_terms": sections_data["relevant_terms"],
            "total_chunks_processed": len(ranked_items),
            "processing_stats": {
                "unique_sections_found": len(set(item['chunk']['metadata']['section_title'] for item in ranked_items if item.get('chunk') and item['chunk'].get('metadata'))),
                "avg_chunk_score": float(round(np.mean([item.get('blended_score', 0) for item in ranked_items]), 3)) if ranked_items else 0.0
            }
        }