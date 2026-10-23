"""Lexical similarity utilities for predicate and constant names.

Provides CamelCase tokenization, lightweight stemming, stopword filtering
and Levenshtein-based similarity measures used throughout the alignment
pipeline.
"""

import re
from rapidfuzz.distance import Levenshtein


# =========================================================
# Stopwords (extended — without negation markers)
# =========================================================
STOP_TOKENS = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "of", "by", "with", "to", "from", "at", "on", "in", "for",
    "into", "onto", "upon", "within", "without",
    # Copulas
    "is", "are", "was", "were", "be", "been", "being",
    # Auxiliaries
    "has", "have", "had", "having",
    "do", "does", "did", "doing",
    # Modals
    "can", "could", "may", "might", "will", "would",
    "shall", "should", "must",
    # Conjunctions (NOT "not" and "no" — kept for negation)
    "and", "or", "but", "nor", "so", "yet",
    # Determiners
    "this", "that", "these", "those",
    "some", "any", "all", "each", "every",
    "both", "either", "neither",
    # Pronouns
    "it", "its", "he", "she", "they", "them", "we", "us",
    # Common adverbs
    "very", "too", "also", "just", "only",
}

PREDICATE_PREFIXES = ("is", "has", "get", "be", "do", "can", "may", "will")
STEM_SUFFIXES = ("ly", "es", "s", "ed", "er", "ing")


def normalize(name):
    """Lowercase a name and strip underscores and hyphens."""
    name = name.lower()
    name = name.replace("_", "")
    name = name.replace("-", "")
    return name

def tokenize(name):
    """Split CamelCase into tokens."""
    tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+', name)
    return [token.lower() for token in tokens]


def stem(token):
    """Strip the first matching suffix from STEM_SUFFIXES (min stem length 3)."""
    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def predicate_tokens(name):
    """Tokenize + remove stopwords + stem."""
    return [
        stem(token)
        for token in tokenize(name)
        if token not in STOP_TOKENS
    ]


def list_similarity(tokens_left, tokens_right):
    """Symmetric average of best matches."""
    if not tokens_left or not tokens_right:
        return 0.0

    left = sum(
        max(Levenshtein.normalized_similarity(t, o) for o in tokens_right)
        for t in tokens_left
    )
    right = sum(
        max(Levenshtein.normalized_similarity(t, o) for o in tokens_left)
        for t in tokens_right
    )
    return (left / len(tokens_left) + right / len(tokens_right)) / 2


def sorted_tokens_similarity(tokens_left, tokens_right):
    """Order-free similarity: compare the tokens after sorting them."""
    left = "".join(sorted(tokens_left))
    right = "".join(sorted(tokens_right))
    return Levenshtein.normalized_similarity(left, right)


def name_similarity(left, right):
    """Similarity between two predicate names."""
    tokens_left = predicate_tokens(left)
    tokens_right = predicate_tokens(right)

    if not tokens_left or not tokens_right:
        return 0.0

    char_similarity = Levenshtein.normalized_similarity(
        "".join(tokens_left), "".join(tokens_right)
    )
    token_similarity_score = list_similarity(tokens_left, tokens_right)
    return max(char_similarity, token_similarity_score)


# =========================================================
# Constants
# =========================================================
def trailing_index(name):
    """Return the trailing integer of a name (e.g. 'person2' -> '2'), else None."""
    match = re.search(r'(\d+)$', name)
    return match.group(1) if match else None


def different_indices(left, right):
    """True if both names end with a trailing index and the indices differ."""
    left_index = trailing_index(left)
    right_index = trailing_index(right)
    return (
        left_index is not None
        and right_index is not None
        and left_index != right_index
    )


def constant_similarity(left, right):
    """Best of character-level, order-free and token-level Levenshtein similarities."""
    char_similarity = Levenshtein.normalized_similarity(
        normalize(left), normalize(right)
    )
    tokens_left = tokenize(left)
    tokens_right = tokenize(right)
    order_free_similarity = sorted_tokens_similarity(tokens_left, tokens_right)
    token_similarity_score = list_similarity(tokens_left, tokens_right)
    return max(char_similarity, order_free_similarity, token_similarity_score)
