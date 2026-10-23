"""Construction of predicate and constant profiles from an enriched DAG.

Each profile aggregates, for one signature (or constant), the claims, FOL
occurrences and claim IDs in which it appears.
"""

from collections import defaultdict


def build_predicates_profiles(enriched_dag):
    """Aggregate per-signature profiles (claims, FOL occurrences, claim IDs).

    Returns:
        sorted list of profile dicts keyed by (predicate, arity).
    """
    profiles = defaultdict(lambda: {
        "claims": [],
        "fol_occurrences": [],
        "claim_ids": []
    })

    for node in enriched_dag:
        for item in node.get("predicates", []):
            key = (item["predicate"], item["arity"])
            profile = profiles[key]

            if node["claim"] not in profile["claims"]:
                profile["claims"].append(node["claim"])

            if node["fol"] not in profile["fol_occurrences"]:
                profile["fol_occurrences"].append(node["fol"])

            if node["claim_id"] not in profile["claim_ids"]:
                profile["claim_ids"].append(node["claim_id"])


    return [
        {
            "predicate": name,
            "arity": arity,
            "signature": (name, arity),
            **data
        }
        for (name, arity), data in sorted(profiles.items())
    ]


def build_constant_profiles(enriched_dag):
    """Aggregate per-constant profiles (claims, FOL occurrences, claim IDs).

    Returns:
        sorted list of profile dicts keyed by constant name.
    """
    profiles = defaultdict(lambda: {
        "claims": [],
        "fol_occurrences": [],
        "claim_ids": []
    })

    for node in enriched_dag:
        for key in node.get("constants", []):

            profile = profiles[key]

            if node["claim"] not in profile["claims"]:
                profile["claims"].append(node["claim"])

            if node["fol"] not in profile["fol_occurrences"]:
                profile["fol_occurrences"].append(node["fol"])

            if node["claim_id"] not in profile["claim_ids"]:
                profile["claim_ids"].append(node["claim_id"])


    return [
        {
            "constant": constant,
            **data
        }
        for constant, data in sorted(profiles.items())
    ]
