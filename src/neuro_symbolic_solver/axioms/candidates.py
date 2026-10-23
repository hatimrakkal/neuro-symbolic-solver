"""Generation of candidate FOL bridge axioms.

For a classified predicate-pair relation, enumerates implication and
equivalence formulations (same arity) and projection formulations
(unary/binary mixes). For clusters, enumerates conjunctive/disjunctive
formulations between left/right subsets of predicates.
"""

from itertools import combinations

from ..lexicon.similarity import predicate_tokens
from ..lexicon.alpha import contains


def _atom(predicate, arity):
    """Render a placeholder atom for a predicate of arity 1 or 2 (else None)."""
    if arity == 1:
        return f"{predicate}(x)"
    if arity == 2:
        return f"{predicate}(x, y)"
    return None


def _quantifier_prefix(arities):
    """Quantifier prefix covering all variables used by the given arities."""
    if any(a == 2 for a in arities):
        return "all x. all y. "
    return "all x. "



def _valid_cluster_relation(left_names, right_names, theta):
    """
    Check that the lexical relation between the union of left tokens
    and the union of right tokens is EQUIVALENT, LCR, or RCL.

    Returns True if at least one containment direction holds.
    """
    left_tokens = set()
    for n in left_names:
        left_tokens.update(predicate_tokens(n))

    right_tokens = set()
    for n in right_names:
        right_tokens.update(predicate_tokens(n))

    if not left_tokens or not right_tokens:
        return False

    left_in_right = contains(left_tokens, right_tokens, theta)
    right_in_left = contains(right_tokens, left_tokens, theta)

    return left_in_right or right_in_left


def _mutually_unrelated(names, theta):
    """
    Check that no two names in the list are lexically related
    (no LCR, RCL, or EQ between any pair).

    This prevents generating redundant conjunctions/disjunctions
    where one predicate is already contained in another.
    """
    if len(names) < 2:
        return True

    for i, j in combinations(range(len(names)), 2):
        tokens_i = set(predicate_tokens(names[i]))
        tokens_j = set(predicate_tokens(names[j]))

        if not tokens_i or not tokens_j:
            continue

        left_in_right = contains(tokens_i, tokens_j, theta)
        right_in_left = contains(tokens_j, tokens_i, theta)

        if left_in_right or right_in_left:
            return False

    return True


def build_candidates(P, arity_P, Q, arity_Q, relation):
    """Dispatch candidate generation by arity combination and relation."""
    if arity_P == arity_Q:
        return _candidates_same_arity(P, Q, arity_P, relation)
    if arity_P == 1 and arity_Q == 2:
        return _candidates_unary_binary(P, Q, relation)
    if arity_P == 2 and arity_Q == 1:
        return _candidates_binary_unary(P, Q, relation)
    return []


def _candidates_same_arity(P, Q, arity, relation):
    """Enumerate axiom candidates for two predicates of the same arity."""
    if arity == 1:
        if relation == "EQUIVALENT":
            return [f"all x. ({P}(x) <-> {Q}(x))"]
        if relation == "NEGATION":
            return [f"all x. ({P}(x) -> -{Q}(x))"]
        return [
            f"all x. ({P}(x) -> {Q}(x))",
            f"all x. ({Q}(x) -> {P}(x))",
            f"all x. ({P}(x) <-> {Q}(x))",
            f"all x. ({P}(x) -> -{Q}(x))",
        ]

    if arity == 2:
        if relation == "EQUIVALENT":
            return [
                f"all x. all y. ({P}(x, y) <-> {Q}(x, y))",
                f"all x. all y. ({P}(x, y) <-> {Q}(y, x))",
            ]
        if relation == "NEGATION":
            return [
                f"all x. all y. ({P}(x, y) -> -{Q}(x, y))",
                f"all x. all y. ({P}(x, y) -> -{Q}(y, x))",
            ]
        return [
            f"all x. all y. ({P}(x, y) -> {Q}(x, y))",
            f"all x. all y. ({P}(x, y) -> {Q}(y, x))",
            f"all x. all y. ({Q}(x, y) -> {P}(x, y))",
            f"all x. all y. ({Q}(x, y) -> {P}(y, x))",
            f"all x. all y. ({P}(x, y) <-> {Q}(x, y))",
            f"all x. all y. ({P}(x, y) <-> {Q}(y, x))",
        ]

    return []


def _candidates_unary_binary(P, Q, relation):
    """Enumerate axiom candidates for a unary left and binary right predicate."""
    if relation == "EQUIVALENT":
        return [
            f"all x. ({P}(x) <-> exists y. {Q}(x, y))",
            f"all x. ({P}(x) <-> exists y. {Q}(y, x))",
        ]
    return [
        f"all x. ({P}(x) -> exists y. {Q}(x, y))",
        f"all x. ({P}(x) -> exists y. {Q}(y, x))",
        f"all x. (exists y. {Q}(x, y) -> {P}(x))",
        f"all x. (exists y. {Q}(y, x) -> {P}(x))",
        f"all x. ({P}(x) <-> exists y. {Q}(x, y))",
        f"all x. ({P}(x) <-> exists y. {Q}(y, x))",
    ]


def _candidates_binary_unary(P, Q, relation):
    """Enumerate axiom candidates for a binary left and unary right predicate."""
    if relation == "EQUIVALENT":
        return [
            f"all x. ({Q}(x) <-> exists y. {P}(x, y))",
            f"all x. ({Q}(x) <-> exists y. {P}(y, x))",
        ]
    return [
        f"all x. all y. ({P}(x, y) -> {Q}(x))",
        f"all x. all y. ({P}(x, y) -> {Q}(y))",
        f"all x. ({Q}(x) -> exists y. {P}(x, y))",
        f"all x. ({Q}(x) -> exists y. {P}(y, x))",
        f"all x. ({Q}(x) <-> exists y. {P}(x, y))",
        f"all x. ({Q}(x) <-> exists y. {P}(y, x))",
    ]


def build_cluster_candidates(
    cluster,
    max_subset_size=2,
    theta=0.8,
    dag_has_binary=False
):
    """Enumerate conjunctive/disjunctive axiom candidates for a cluster.

    Left and right sides are subsets of the cluster (up to max_subset_size),
    filtered for lexical validity, combined with -> and <-> over AND/OR
    join patterns and deduplicated.
    """
    
    n = len(cluster)
    if n < 2:
        return []

    names = [c["predicate"] for c in cluster]
    arities = [c["arity"] for c in cluster]

    if dag_has_binary:
        q = "all x. all y. "
    else:
        q = _quantifier_prefix(arities)

    indices = list(range(n))
    candidates = []

    for size_L in range(1, min(max_subset_size, n - 1) + 1):
        for L in combinations(indices, size_L):
            remaining = [i for i in indices if i not in L]

            left_names = [names[i] for i in L]
            if not _mutually_unrelated(left_names, theta):
                continue

            for size_R in range(1, min(max_subset_size, len(remaining)) + 1):
                for R in combinations(remaining, size_R):

                    right_names = [names[i] for i in R]

                    if not _mutually_unrelated(right_names, theta):
                        continue

                    if not _valid_cluster_relation(left_names, right_names, theta):
                        continue

                    left_atoms = []
                    valid = True
                    for i in L:
                        a = _atom(names[i], arities[i])
                        if a is None:
                            valid = False
                            break
                        left_atoms.append(a)

                    if not valid:
                        continue

                    right_atoms = []
                    for i in R:
                        a = _atom(names[i], arities[i])
                        if a is None:
                            valid = False
                            break
                        right_atoms.append(a)

                    if not valid:
                        continue

                    and_L = " & ".join(left_atoms)
                    or_L = " | ".join(left_atoms)
                    and_R = " & ".join(right_atoms)
                    or_R = " | ".join(right_atoms)

                    candidates.append(f"{q}(({and_L}) -> ({and_R}))")
                    candidates.append(f"{q}(({and_L}) -> ({or_R}))")
                    candidates.append(f"{q}(({or_L}) -> ({and_R}))")
                    candidates.append(f"{q}(({or_L}) -> ({or_R}))")
                    candidates.append(f"{q}(({and_R}) -> ({and_L}))")
                    candidates.append(f"{q}(({and_R}) -> ({or_L}))")
                    candidates.append(f"{q}(({or_R}) -> ({and_L}))")
                    candidates.append(f"{q}(({or_R}) -> ({or_L}))")
                    candidates.append(f"{q}(({and_L}) <-> ({and_R}))")
                    candidates.append(f"{q}(({and_L}) <-> ({or_R}))")
                    candidates.append(f"{q}(({or_L}) <-> ({and_R}))")
                    candidates.append(f"{q}(({or_L}) <-> ({or_R}))")

    return list(dict.fromkeys(candidates))
