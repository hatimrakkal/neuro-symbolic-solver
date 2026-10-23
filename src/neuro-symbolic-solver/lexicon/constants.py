"""Classification of constant-pair relations based on lexical similarity.

Pairs whose normalized Levenshtein similarity exceeds a threshold (and that
do not carry different trailing indices, e.g. person1 vs person2) are marked
SAME_ENTITY; all other pairs are UNRELATED.
"""

from .similarity import normalize, constant_similarity, different_indices


def classify_constants_relation(candidates, similarity_threshold=0.85):
    """
    Classify constant candidate pairs as SAME_ENTITY or UNRELATED.

    Args:
        candidates: list of (left_profile, right_profile) dicts,
                    each exposing a "constant" key.
        similarity_threshold: minimum constant similarity for SAME_ENTITY.

    Returns:
        list of dicts with left/right constants, normalized forms,
        similarity, relation and confidence.
    """
    classified_pairs = []

    for left_constant, right_constant in candidates:

        normalized_left = normalize(left_constant["constant"])
        normalized_right = normalize(right_constant["constant"])


        similarity = constant_similarity(left_constant["constant"], right_constant["constant"])

        if similarity >= similarity_threshold and not different_indices(left_constant["constant"], right_constant["constant"]):
            relation = "SAME_ENTITY"
        else:
            relation = "UNRELATED"

        classified_pairs.append({
            "left": left_constant["constant"],
            "right": right_constant["constant"],
            "normalized_left": normalized_left,
            "normalized_right": normalized_right,
            "levenshtein_similarity": float(similarity),
            "relation": relation,
            "confidence": float(similarity),
        })

    return classified_pairs
