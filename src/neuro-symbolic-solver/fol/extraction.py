"""Extraction of predicates and constants from parsed FOL formulas.

Builds on NLTK's semantic parser to collect predicate signatures, enrich
translated DAG nodes, compute co-occurrence ("cannot-link") pairs within a
claim, and generate candidate predicate/constant pairs across profiles.
"""

from nltk.sem.logic import Expression, ApplicationExpression
from itertools import combinations

def extract_predicates(fol_list):
    """Collect the distinct (predicate, arity) signatures used in FOL formulas.

    Args:
        fol_list: list of FOL strings.

    Returns:
        sorted list of dicts: {"predicate", "arity", "signature"}.
    """
    signatures = set()

    def visit(node, predicate_names):
        if isinstance(node, ApplicationExpression):
            name = node.pred.variable.name

            if name in predicate_names:
                signatures.add((name, len(node.args)))

            for argument in node.args:
                visit(argument, predicate_names)
            return

        for attribute in ("term", "first", "second"):
            child = getattr(node, attribute, None)
            if child is not None:
                visit(child, predicate_names)

    for fol in fol_list:
        try:
            expr = Expression.fromstring(fol)
        except Exception:
            continue
        predicate_names = {
            predicate.name for predicate in expr.predicates()
        }
        visit(expr, predicate_names)

    return [
        {
            "predicate": name,
            "arity": arity,
            "signature": f"{name}/{arity}"
        }
        for name, arity in sorted(signatures)
    ]


def extract_constants(fol_list):
    """Collect the sorted set of constant names appearing in FOL formulas."""
    all_constants = set()

    for fol in fol_list:
        try:
            expression = Expression.fromstring(fol)
            constants = {constant.name for constant in expression.constants()}
            all_constants.update(constants)

        except Exception as error:
            print(f"Error : {error}")

    return sorted(all_constants)


def enrich_nodes(translated_dag):
    """Attach extracted predicates and constants to every node of a translated DAG."""
    enriched = []

    for node in translated_dag:
        fol = node.get("fol")

        if not fol:
            enriched.append({
                **node,
                "predicates": [],
                "constants": []
            })
            continue

        predicates = extract_predicates([fol])
        constants = extract_constants([fol])

        enriched.append({
            **node,
            "predicates": predicates,
            "constants": constants
        })

    return enriched


def build_cooccurence_pairs(enriched_dag):
    """Compute predicate/constant pairs co-occurring inside the same claim.

    These pairs act as "cannot links": signatures appearing together in one
    claim are excluded from cross-claim candidate generation.

    Returns:
        (predicate_pairs, constant_pairs) as sets of sorted tuples.
    """
    predicate_pairs = set()
    constant_pairs = set()
    for node in enriched_dag:
        signatures = sorted({(p["predicate"], p["arity"]) for p in node.get("predicates", [])})
        constants = sorted({c for c in node.get("constants", [])})

        for left, right in combinations(signatures, 2):
            predicate_pairs.add(tuple(sorted((left, right))))

        for left, right in combinations(constants, 2):
            constant_pairs.add(tuple(sorted((left, right))))

    return predicate_pairs, constant_pairs


def generate_predicate_pairs(predicate_profiles, cannot_links):
    """Generate candidate predicate pairs across profiles, skipping cannot links."""
    candidates = []

    for left, right in combinations(predicate_profiles, 2):

        signature_left = (left["predicate"], left["arity"])
        signature_right = (right["predicate"], right["arity"])

        pair_key = tuple(sorted((signature_left, signature_right)))

        if pair_key in cannot_links:
            continue

        candidates.append((left, right))

    return candidates


def generate_constant_pairs(constant_profiles, cannot_links):
    """Generate candidate constant pairs across profiles, skipping cannot links."""
    candidates = []

    for left, right in combinations(constant_profiles, 2):

        pair_key = tuple(sorted((left["constant"], right["constant"])))

        if pair_key in cannot_links:
            continue

        candidates.append((left, right))

    return candidates
