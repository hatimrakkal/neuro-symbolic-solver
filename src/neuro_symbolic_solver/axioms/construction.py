"""Main orchestrator: builds bridge axioms from translated DAGs.

Combines node enrichment, profile construction, pair classification
(Rule 1), cluster building (Rule 2), Strategy C candidate processing and
constant alignment into a single entry point, `build_bridge_axioms`.
"""

from ..fol.extraction import (
    enrich_nodes,
    build_cooccurence_pairs,
    generate_predicate_pairs,
    generate_constant_pairs,
)
from ..fol.profiles import build_predicates_profiles, build_constant_profiles
from ..lexicon.similarity import predicate_tokens
from ..lexicon.alpha import classify
from ..lexicon.constants import classify_constants_relation
from .cluster_expansion import expand_cluster_with_existing_subpredicates
from .pipeline import process_all_candidates


# =========================================================
# Helpers
# =========================================================
def _has_partial_overlap(name1, name2):
    """
    Return True if the two predicates share at least one token.
    """
    tokens1 = set(predicate_tokens(name1))
    tokens2 = set(predicate_tokens(name2))
    if not tokens1 or not tokens2:
        return False
    return bool(tokens1 & tokens2)


def _qualifies_for_cluster(cand):
    """
    A pair qualifies for clustering if:
      - it has different arities, OR
      - it shares at least one token (partial overlap).
    """
    left_name, left_arity = cand["left"]
    right_name, right_arity = cand["right"]

    if left_arity != right_arity:
        return True

    return _has_partial_overlap(left_name, right_name)


def align_constants(classified_constants):
    """
    Build equality axioms from classified constant pairs.
    Only SAME_ENTITY pairs produce axioms.
    """
    fols = []
    for constant in classified_constants:
        if constant["relation"] == "SAME_ENTITY":
            fols.append(f"{constant['left']} = {constant['right']}")
    return fols


# =========================================================
# Cluster building
# =========================================================
def build_clusters_from_classified(
    classified_pairs,
    profile_by_signature,
    max_cluster_size=6,
):
    """
    Build and expand clusters from classified pairs.

    Steps:
      1. Collect signatures involved in qualifying pairs.
      2. Group by token overlap (single pass, transitive).
      3. Expand each cluster with existing sub-predicates.

    Args:
        classified_pairs: list from alpha.classify
        profile_by_signature: dict {(name, arity): profile}
        max_cluster_size: cap on the cluster size after expansion

    Returns:
        list of clusters; each cluster is a list of dicts
        with keys 'predicate' and 'arity'.
    """
    # --- Step 1: collect signatures that qualify ---
    involved = set()
    for cand in classified_pairs:
        if not _qualifies_for_cluster(cand):
            continue
        left_name, left_arity = cand["left"]
        right_name, right_arity = cand["right"]
        involved.add((left_name, left_arity))
        involved.add((right_name, right_arity))

    if not involved:
        return []

    # --- Step 2: group by token overlap ---
    signatures = sorted(involved)
    tokens_by_sig = {
        sig: set(predicate_tokens(sig[0])) for sig in signatures
    }

    clusters = []
    assigned = set()

    for sig in signatures:
        if sig in assigned:
            continue

        cluster = [sig]
        assigned.add(sig)

        for other in signatures:
            if other in assigned:
                continue
            if tokens_by_sig[sig] & tokens_by_sig[other]:
                cluster.append(other)
                assigned.add(other)

        cluster_dicts = [
            {"predicate": name, "arity": arity}
            for name, arity in cluster
        ]

        # --- Step 3: collect occurrences for expansion ---
        occurrences = []
        for item in cluster_dicts:
            profile = profile_by_signature.get(
                (item["predicate"], item["arity"])
            )
            if profile:
                occurrences.extend(profile.get("fol_occurrences", []))

        # --- Step 4: expand with sub-predicates ---
        expanded = expand_cluster_with_existing_subpredicates(
            cluster_dicts,
            occurrences,
            max_cluster_size=max_cluster_size,
        )

        clusters.append(expanded)

    return [c for c in clusters if len(c) >= 2]


# =========================================================
# Main orchestrator
# =========================================================
def build_bridge_axioms(
    translated_dags,
    judge=None,
    theta=0.8,
    constant_similarity_threshold=0.85,
    base_theory="",
    timeout=15,
    workers=4,
    group_size=5,
    max_rounds=3,
    max_subset_size=2,
    max_cluster_size=6,
    verbose=False,
):
    """
    Build bridge axioms for a list of translated DAGs.

    Pipeline per DAG:
      1. Enrich and build profiles.
      2. Compute dag_has_binary.
      3. Generate predicate pairs and classify them (Rule 1 decision).
      4. Build clusters from classified pairs (Rule 2 input).
      5. Process all candidates (Rule 1 + Rule 2 + Strategy C).
      6. Align constants.
      7. Combine and deduplicate.

    Args:
        translated_dags: list of DAGs (each is a list of nodes)
        judge: optional LLMJudge instance (None = deterministic fallback)
        theta: lexical similarity threshold (0-1)
        constant_similarity_threshold: threshold for constant alignment
        base_theory: additional FOL axioms (string, no trailing dot)
        timeout: Prover9 timeout per call (seconds)
        workers: number of threads for Prover9
        group_size: LLM group size for cascade
        max_rounds: LLM cascade max depth
        max_subset_size: max size of L and R in Rule 2
        max_cluster_size: cap on cluster size after expansion
        verbose: print progress

    Returns:
        dict: dag_id -> list of axiom strings
    """
    axioms_by_dag = {}

    for dag_id, dag in enumerate(translated_dags):

        # --- 1. Enrich and profiles ---
        enriched = enrich_nodes(dag)
        predicate_profiles = build_predicates_profiles(enriched)
        constant_profiles = build_constant_profiles(enriched)

        # --- 2. dag_has_binary ---
        dag_has_binary = any(
            p["arity"] == 2 for p in predicate_profiles
        )

        # --- 3. Rule 1: pairs and classification ---
        cannot_links = build_cooccurence_pairs(enriched)
        predicate_candidates = generate_predicate_pairs(
            predicate_profiles, cannot_links[0]
        )
        classified_pairs = classify(predicate_candidates, theta=theta)

        profile_by_signature = {
            (p["predicate"], p["arity"]): p
            for p in predicate_profiles
        }

        # --- 4. Rule 2: clusters ---
        clusters = build_clusters_from_classified(
            classified_pairs,
            profile_by_signature,
            max_cluster_size=max_cluster_size,
        )

        # --- 5. Base theory for this DAG ---
        dag_theory = "\n".join(
            node.get("fol", "")
            for node in dag
            if node.get("fol")
        )
        combined_theory = (base_theory + "\n" + dag_theory).strip()

        # --- 6. Process all candidates ---
        result = process_all_candidates(
            classified_pairs=classified_pairs,
            base_theory=combined_theory,
            clusters=clusters,
            judge=judge,
            profile_by_signature=profile_by_signature,
            timeout=timeout,
            workers=workers,
            group_size=group_size,
            max_rounds=max_rounds,
            max_subset_size=max_subset_size,
            dag_has_binary=dag_has_binary,
            theta=theta,
            verbose=verbose,
        )

        # --- 7. Constant alignment ---
        constant_candidates = generate_constant_pairs(
            constant_profiles, cannot_links[1]
        )
        classified_constants = classify_constants_relation(
            constant_candidates,
            similarity_threshold=constant_similarity_threshold,
        )
        equality_axioms = align_constants(classified_constants)

        # --- 8. Combine and deduplicate ---
        combined = (
            result["pair_axioms"]
            + result["cluster_axioms"]
            + equality_axioms
        )
        axioms_by_dag[dag_id] = list(dict.fromkeys(combined))

    return axioms_by_dag
