from nltk.corpus import wordnet
import logging

logger = logging.getLogger(__name__)

def get_synonyms(word):
    """
    Retrieve synonyms for a given word using WordNet, filtering for nouns, verbs, and adjectives.
    Returns up to 3 relevant synonyms, prioritizing travel-related terms.
    """
    try:
        synonyms = set()
        # Get synsets for the word, limiting to nouns (n), verbs (v), and adjectives (a)
        for synset in wordnet.synsets(word, pos=[wordnet.NOUN, wordnet.VERB, wordnet.ADJ]):
            # Extract lemma names (synonyms) from the synset
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                # Filter for single words or short phrases, avoid duplicates
                if len(synonym.split()) <= 2 and synonym != word.lower():
                    synonyms.add(synonym)
        
        # Prioritize travel-related synonyms (optional heuristic)
        travel_keywords = ['tour', 'journey', 'activity', 'event', 'plan', 'itinerary', 'group', 'travel', 'voyage']
        prioritized = [s for s in synonyms if any(kw in s for kw in travel_keywords)]
        others = [s for s in synonyms if s not in prioritized]
        
        # Combine and limit to 3 synonyms
        result = (prioritized + others)[:3]
        logger.debug(f"Synonyms for '{word}': {result}")
        return result
    
    except Exception as e:
        logger.warning(f"Failed to get synonyms for '{word}': {e}")
        return []