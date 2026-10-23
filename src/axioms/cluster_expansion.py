"""Expansion of predicate clusters with existing sub-predicates.

A predicate observed in the FOL occurrences joins a cluster when its token
signature is a strict subset of a cluster member's signature, up to a
maximum cluster size.
"""

from ..fol.extraction import extract_predicates
from ..lexicon.similarity import predicate_tokens

def collect_predicates_from_occurrences(occurrences):
    """Collect the distinct (predicate, arity) signatures from FOL occurrences."""
    return [
        (item["predicate"], item["arity"])
        for item in extract_predicates(occurrences)
    ]


def _tokens_signature(predicate):
    """Token set signature of a predicate name."""
    return set(predicate_tokens(predicate))


def _is_strict_subset(candidate_tokens, reference_tokens):
    """
    Return True if candidate_tokens is a strict subset of reference_tokens.
    """
    if not candidate_tokens or not reference_tokens:
        return False
    return candidate_tokens < reference_tokens


def expand_cluster_with_existing_subpredicates(
    cluster,
    occurrences,
    max_cluster_size=8,
):
    """Add observed sub-predicates (strict token subsets) to a cluster.

    Args:
        cluster: list of dicts with "predicate" and "arity" keys.
        occurrences: FOL strings in which sub-predicates may be found.
        max_cluster_size: cap on the final cluster size.

    Returns:
        the expanded cluster (original members first, additions after).
    """
    
    existing = set()
    cluster_tokens = {}
    for item in cluster:
        sig = (item["predicate"], item["arity"])
        existing.add(sig)
        cluster_tokens[sig] = _tokens_signature(item["predicate"])

    
    available = collect_predicates_from_occurrences(occurrences)

    additions = []
    for (name, arity) in available:
        sig = (name, arity)
        if sig in existing:
            continue

        candidate_tokens = _tokens_signature(name)
        if not candidate_tokens:
            continue

        for ref_sig, ref_tokens in cluster_tokens.items():
            if _is_strict_subset(candidate_tokens, ref_tokens):
                additions.append({"predicate": name, "arity": arity})
                break

    seen = set(existing)
    unique_additions = []
    for item in additions:
        sig = (item["predicate"], item["arity"])
        if sig not in seen:
            seen.add(sig)
            unique_additions.append(item)

    current_size = len(cluster)
    remaining = max_cluster_size - current_size

    if remaining <= 0:
        return list(cluster)

    unique_additions = unique_additions[:remaining]

    return list(cluster) + unique_additions
