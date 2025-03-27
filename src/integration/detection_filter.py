import logging
from typing import List, Dict, Any
from nltk.corpus import wordnet as wn

logger = logging.getLogger("AIVOL.integration.detection_filter")


def get_synonyms(word: str) -> set:
    """
    Get a set of synonyms for a given word using WordNet.

    Args:
        word (str): The target word.

    Returns:
        set: A set of synonym strings.
    """
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms


def filter_detections(query: dict, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter detection results based on query information. This function uses both exact
    matching and synonym matching (via NLTK WordNet) to catch similar target words.

    Args:
        query (dict): Query information with keys like "name", "color", "location".
        detections (list): List of detection dictionaries, each with keys:
            - "label": detected class name,
            - "confidence": confidence score,
            - "bbox": bounding box coordinates [x1, y1, x2, y2].

    Returns:
        list: A list containing the best matching detection(s), e.g., the one with the highest confidence.
    """
    filtered = []
    query_name = query.get("name", "").lower().strip()

    # Get synonyms for the query name from WordNet
    synonyms = get_synonyms(query_name)
    synonyms.add(query_name)  # Ensure the query itself is included
    logger.info("Synonyms for '%s': %s", query_name, synonyms)

    # Filter detections: check if detection label is an exact match or in the synonyms
    for det in detections:
        label = det.get("label", "").lower().strip()
        if label in synonyms:
            filtered.append(det)

    # If no detection is found, try a more lenient match (substring match)
    if not filtered:
        for det in detections:
            label = det.get("label", "").lower().strip()
            if query_name in label or label in query_name:
                filtered.append(det)

    if filtered:
        best_detection = max(filtered, key=lambda x: x.get("confidence", 0))
        return [best_detection]

    return filtered
