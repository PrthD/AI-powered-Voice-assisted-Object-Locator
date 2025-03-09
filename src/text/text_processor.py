"""
Module: text_processor.py
Purpose: Process and analyze text commands for object queries using NLP libraries.
Provides advanced processing with spaCy (if available) and a fallback using NLTK.
Also sets up required NLP resources.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

# Advanced NLP libraries
import nltk
import spacy
from spacy.cli import download as spacy_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Set up module-level logger
logger = logging.getLogger("AIVOL.text_processor")

# --- NLP Setup Functions ---


def setup_nltk_resources() -> bool:
    """
    Download required NLTK resources.

    Returns:
        bool: True if all resources downloaded successfully, False otherwise.
    """
    try:
        resources = ['punkt', 'stopwords',
                     'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
        logger.info("NLTK resources downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        return False


def setup_spacy_model() -> Optional[spacy.language.Language]:
    """
    Download and load the spaCy model 'en_core_web_sm'. If not found, auto-download it.

    Returns:
        Optional[spacy.language.Language]: The loaded spaCy model or None if setup fails.
    """
    try:
        try:
            model = spacy.load('en_core_web_sm')
            logger.info("spaCy model 'en_core_web_sm' is already installed.")
            return model
        except OSError:
            logger.info(
                "spaCy model not found. Installing 'en_core_web_sm'...")
            spacy_download('en_core_web_sm')
            model = spacy.load('en_core_web_sm')
            logger.info("spaCy model installed successfully.")
            return model
    except Exception as e:
        logger.error(f"Error setting up spaCy model: {e}")
        return None


# Run NLP setup once during module initialization.
_nltk_success = setup_nltk_resources()
nlp = setup_spacy_model()

# --- End NLP Setup ---

# Load stopwords and initialize lemmatizer
try:
    stop_words: Set[str] = set(stopwords.words('english'))
    # Add custom stopwords
    custom_stopwords = {'the', 'a', 'an', 'please', 'can', 'you',
                        'me', 'i', 'my', 'find', 'locate', 'where', 'is', 'are'}
    stop_words.update(custom_stopwords)
except Exception as e:
    logger.warning(f"Error loading stopwords: {e}. Using basic stopword list.")
    stop_words = {'the', 'a', 'an', 'please', 'can', 'you',
                  'me', 'i', 'my', 'find', 'locate', 'where', 'is', 'are'}

lemmatizer = WordNetLemmatizer()

# Descriptor constants
COLORS: Set[str] = {
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange',
    'brown', 'pink', 'gray', 'grey', 'silver', 'gold', 'transparent', 'dark', 'light'
}

LOCATION_PREPOSITIONS: Set[str] = {
    'on', 'in', 'at', 'near', 'by', 'under', 'above', 'behind', 'next', 'to',
    'beside', 'between', 'inside', 'outside', 'around', 'across', 'over', 'below'
}

SIZE_DESCRIPTORS: Set[str] = {
    'big', 'small', 'large', 'tiny', 'huge', 'little', 'medium', 'giant',
    'miniature', 'massive', 'enormous', 'microscopic', 'tall', 'short', 'wide', 'narrow'
}

SHAPE_DESCRIPTORS: Set[str] = {
    'round', 'square', 'rectangular', 'circular', 'oval', 'triangular', 'flat',
    'curved', 'straight', 'cylindrical', 'spherical', 'cubic', 'pyramid', 'cone'
}

MATERIAL_DESCRIPTORS: Set[str] = {
    'wooden', 'metal', 'plastic', 'glass', 'ceramic', 'paper', 'cardboard',
    'steel', 'iron', 'aluminum', 'cloth', 'fabric', 'leather', 'cotton', 'wool'
}

# Precompile regex patterns for efficiency in is_valid_object_query
QUERY_PATTERNS: List[re.Pattern] = [
    # "Where is the..."
    re.compile(r'where.*(?:is|are).*', re.IGNORECASE),
    # "Find the..." or "Locate the..."
    re.compile(r'(?:find|locate).*', re.IGNORECASE),
    re.compile(r'(?:can|could).*(?:find|locate|see|spot).*',
               re.IGNORECASE),  # "Can you find..."
    # "I'm looking for..."
    re.compile(r'looking.*for.*', re.IGNORECASE),
    re.compile(r'help.*(?:find|locate).*',
               re.IGNORECASE),     # "Help me find..."
    # "Search for..."
    re.compile(r'search.*for.*', re.IGNORECASE),
    # "Show me the..."
    re.compile(r'show.*(?:me|the).*', re.IGNORECASE)
]


class ObjectInfo:
    """Class to store information about the target object."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: Optional[str] = name
        self.color: Optional[str] = None
        self.size: Optional[str] = None
        self.shape: Optional[str] = None
        self.material: Optional[str] = None
        self.location: Optional[str] = None
        self.other_qualifiers: List[str] = []

    def __str__(self) -> str:
        parts = []
        if self.color:
            parts.append(f"color: {self.color}")
        if self.size:
            parts.append(f"size: {self.size}")
        if self.shape:
            parts.append(f"shape: {self.shape}")
        if self.material:
            parts.append(f"material: {self.material}")
        if self.location:
            parts.append(f"location: {self.location}")
        if self.other_qualifiers:
            parts.append(f"other: {', '.join(self.other_qualifiers)}")
        qualifiers = ", ".join(parts)
        return f"Object: {self.name or 'Unknown'}" + (f" ({qualifiers})" if qualifiers else "")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            'name': self.name,
            'color': self.color,
            'size': self.size,
            'shape': self.shape,
            'material': self.material,
            'location': self.location,
            'other_qualifiers': ', '.join(self.other_qualifiers)
        }


def get_compound_noun(token: spacy.tokens.Token) -> str:
    """
    Construct a compound noun phrase by joining left-side compound tokens with the head noun.

    Args:
        token (spacy.tokens.Token): The head noun token.

    Returns:
        str: The full compound noun phrase.
    """
    compounds = [
        child.text for child in token.lefts if child.dep_ == 'compound']
    compounds.append(token.text)
    return " ".join(compounds)


def extract_location_phrase(doc: spacy.tokens.Doc) -> Optional[str]:
    """
    Extract location phrases from a spaCy document.
    If multiple location phrases are found, return the longest one.

    Args:
        doc (spacy.tokens.Doc): The processed spaCy document.

    Returns:
        Optional[str]: The most specific location phrase if found, otherwise None.
    """
    prep_phrases = []
    for token in doc:
        if token.text.lower() in LOCATION_PREPOSITIONS:
            phrase = [token.text]
            for child in token.children:
                if child.dep_ in ('pobj', 'dobj'):
                    subtree = [t.text for t in child.subtree]
                    phrase.extend(subtree)
                    break
            if len(phrase) > 1:
                prep_phrases.append(' '.join(phrase))

    if not prep_phrases:
        return None

    # Choose the longest phrase by number of words
    longest_phrase = max(prep_phrases, key=lambda p: len(p.split()))
    return longest_phrase


def _process_adjective(adj_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], List[str]]:
    """
    Process an adjective string and determine if it fits known descriptor categories.

    Args:
        adj_text (str): The adjective text.

    Returns:
        Tuple containing (color, size, shape, material, other_qualifiers)
    """
    lemma = lemmatizer.lemmatize(adj_text.lower(), pos='a')
    color = size = shape = material = None
    other = []
    if lemma in COLORS:
        color = lemma
    elif lemma in SIZE_DESCRIPTORS:
        size = lemma
    elif lemma in SHAPE_DESCRIPTORS:
        shape = lemma
    elif lemma in MATERIAL_DESCRIPTORS:
        material = lemma
    else:
        other.append(lemma)
    return color, size, shape, material, other


def extract_object_info_with_spacy(text: str) -> ObjectInfo:
    """
    Extract object information using spaCy's NLP capabilities.

    Args:
        text (str): The user input text.

    Returns:
        ObjectInfo: Extracted information about the object.
    """
    if not nlp:
        logger.warning(
            "spaCy model not available, falling back to basic processing")
        return extract_object_info_basic(text)

    doc = nlp(text.lower())
    object_info = ObjectInfo()

    # Process adjectives to update descriptors
    for token in doc:
        if token.pos_ == 'ADJ':
            c, s, sh, m, other = _process_adjective(token.text)
            if c and not object_info.color:
                object_info.color = c
            if s and not object_info.size:
                object_info.size = s
            if sh and not object_info.shape:
                object_info.shape = sh
            if m and not object_info.material:
                object_info.material = m
            if other:
                object_info.other_qualifiers.extend(other)

    # Extract location information
    object_info.location = extract_location_phrase(doc)

    # Extract nouns as potential object names
    nouns = [token for token in doc if token.pos_ in ('NOUN', 'PROPN')]
    if object_info.location:
        loc_words = set(object_info.location.split())
        nouns = [noun for noun in nouns if noun.text.lower() not in loc_words]

    # Prefer direct objects or subjects and join compound tokens
    target_nouns = [noun for noun in nouns if noun.dep_ in (
        'dobj', 'pobj', 'nsubj')]
    if target_nouns:
        dobjs = [n for n in target_nouns if n.dep_ in ('dobj', 'pobj')]
        chosen = dobjs[0] if dobjs else target_nouns[0]
        object_info.name = get_compound_noun(chosen).lower()
    elif nouns:
        object_info.name = get_compound_noun(nouns[0]).lower()

    return object_info


def extract_object_info_basic(text: str) -> ObjectInfo:
    """
    Fallback method to extract object information using NLTK.

    Args:
        text (str): The user input text.

    Returns:
        ObjectInfo: Extracted information about the object.
    """
    object_info = ObjectInfo()
    tokens: List[str] = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    filtered_tokens = [
        token for token in lemmatized_tokens if token not in stop_words]
    tagged = pos_tag(filtered_tokens)

    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    adjectives = [word for word, tag in tagged if tag.startswith('JJ')]

    for adj in adjectives:
        c, s, sh, m, other = _process_adjective(adj)
        if c and not object_info.color:
            object_info.color = c
        elif s and not object_info.size:
            object_info.size = s
        elif sh and not object_info.shape:
            object_info.shape = sh
        elif m and not object_info.material:
            object_info.material = m
        else:
            object_info.other_qualifiers.append(adj)

    # Extract simple location information
    for i, token in enumerate(tokens):
        if token in LOCATION_PREPOSITIONS and i < len(tokens) - 1:
            end_idx = min(i + 4, len(tokens))
            location_phrase = ' '.join(tokens[i:end_idx])
            object_info.location = location_phrase
            break

    if nouns:
        # For basic processing, simply take the first noun as the object name.
        object_info.name = nouns[0]

    return object_info


def is_valid_object_query(text: str) -> bool:
    """
    Determine whether the text appears to be a valid object query.

    Args:
        text (str): The input text.

    Returns:
        bool: True if it matches any of the known patterns.
    """
    for pattern in QUERY_PATTERNS:
        if pattern.search(text):
            return True
    return False


def process_user_prompt(text: str) -> ObjectInfo:
    """
    Process the user's text input to extract relevant object information.

    Args:
        text (str): The input text.

    Returns:
        ObjectInfo: Extracted object information.
    """
    if not text:
        logger.error("Received empty text input")
        return ObjectInfo()

    logger.info("Processing user prompt: '%s'", text)
    if not is_valid_object_query(text):
        logger.warning(
            "Input text '%s' does not appear to be a valid object query", text)

    try:
        if nlp:
            object_info = extract_object_info_with_spacy(text)
        else:
            object_info = extract_object_info_basic(text)
    except Exception as e:
        logger.error("Error processing prompt: %s", e)
        object_info = ObjectInfo()

    logger.info("Extracted object info: %s", object_info)
    return object_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_queries = [
        "Where is my red measuring cup?",
        "Can you find the book on the table?",
        "Locate the blue chair in the living room",
        "I'm looking for my car keys",
        "Find the black remote near the TV",
        "Show me the large wooden box under the desk",
        "Help me find my glasses on the kitchen counter",
        "Where are the metal scissors in the drawer?",
        "Can you see a small silver ring anywhere?",
        "I need to find my blue notebook with the spiral binding"
    ]

    print("Testing text processor with example queries:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        info = process_user_prompt(query)
        print(f"Result: {info}")
        print(f"Details: {info.to_dict()}")
