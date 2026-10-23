"""Lexical classification of predicate-pair relations (the "alpha" layer).

Classifies pairs of predicate signatures into EQUIVALENT, NEGATION,
LEFT_CONTAINS_RIGHT, RIGHT_CONTAINS_LEFT, PROJECTION or UNRELATED, using
WordNet antonymy, token containment and name similarity.
"""

from rapidfuzz.distance import Levenshtein
from .similarity import predicate_tokens, name_similarity
from nltk.corpus import wordnet as wn


UNRELATED = "UNRELATED"
EQUIVALENT = "EQUIVALENT"
NEGATION = "NEGATION"
LEFT_CONTAINS_RIGHT = "LEFT_CONTAINS_RIGHT"
RIGHT_CONTAINS_LEFT = "RIGHT_CONTAINS_LEFT"
PROJECTION = "PROJECTION"



def _wordnet_antonyms(word):
    """Collect the lowercase WordNet antonym names of a word."""
    antonyms = set()
    for synset in wn.synsets(word.lower()):
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().lower())

    return antonyms


def is_negation_pair(P, Q):
    """True if one predicate name is a WordNet antonym of the other."""
    p_low = P.lower()
    q_low = Q.lower()

    if p_low in _wordnet_antonyms(q_low):
        return True
    if q_low in _wordnet_antonyms(p_low):
        return True

    return False


def contains(left, right, theta):
    """True if every token in `left` has a Levenshtein match >= theta in `right`."""
    return all(
        any(Levenshtein.normalized_similarity(a, b) >= theta for b in right)
        for a in left
    )



def alpha_force(left, right, theta=0.8):
    """Classify two same-arity predicate names into the alpha relation inventory."""
    tokens_left = predicate_tokens(left)
    tokens_right = predicate_tokens(right)

    if not tokens_left or not tokens_right:
        return UNRELATED

    if is_negation_pair(left, right):
        return NEGATION
    
    if contains(tokens_left, tokens_right, theta) and \
       contains(tokens_right, tokens_left, theta):
        return EQUIVALENT

    if contains(tokens_left, tokens_right, theta):
        return LEFT_CONTAINS_RIGHT

    if contains(tokens_right, tokens_left, theta):
        return RIGHT_CONTAINS_LEFT

    return UNRELATED


def classify_diff_arity(left_name, right_name, theta=0.8):
    """Classify a pair with different arities: PROJECTION above the threshold, else UNRELATED."""
    sim = name_similarity(left_name, right_name)
    if sim >= theta:
        return PROJECTION
    return UNRELATED


def classify_predicates_relation(pair, theta=0.8):
    """Build the classification record (relation + similarity) for one predicate pair."""
    left, right = pair
    left_name = left["predicate"]
    left_arity = left["arity"]
    right_name = right["predicate"]
    right_arity = right["arity"]

    if left_arity == right_arity:
        relation = alpha_force(left_name, right_name, theta=theta)
    else:
        relation = classify_diff_arity(left_name, right_name, theta=theta)

    return {
        "left": (left_name, left_arity),
        "right": (right_name, right_arity),
        "relation": relation,
        "similarity": name_similarity(left_name, right_name),
    }


def classify(candidates, theta=0.8):
    """Classify a list of predicate pairs (see classify_predicates_relation)."""
    return [
        classify_predicates_relation(pair, theta=theta)
        for pair in candidates
    ]
