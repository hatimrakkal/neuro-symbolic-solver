"""Strategy C: consistency/usefulness filtering with parallel Prover9,
followed by LLM cascade selection when candidates conflict.

Also hosts the Rule 1 (pair) and Rule 2 (cluster) processing entry points
and the full candidate processing routine.
"""

from itertools import combinations
from ..prover.prover9 import is_theory_consistent, proves
from ..prover.prover9_parallel import (
    filter_consistent_parallel,
    filter_useful_parallel,
)
from .candidates import build_candidates, build_cluster_candidates
from ..llm.llm_selector import select_with_llm_cascade


# =========================================================
# Compatibility check (Strategy C core)
# =========================================================
def are_mutually_exclusive(c1, c2, base_theory, timeout=15):
    """
    Check if two axioms are contradictory.

    Returns True if c1 + c2 makes the theory inconsistent.
    """
    combined = base_theory + "\n" + c1 + "\n" + c2
    return not is_theory_consistent(combined, timeout=timeout)


def are_all_compatible(candidates, base_theory, timeout=15):
    """
    Check if all candidates in the list are mutually compatible.

    Returns True if no pair is mutually exclusive.
    """
    if len(candidates) <= 1:
        return True

    for c1, c2 in combinations(candidates, 2):
        if are_mutually_exclusive(c1, c2, base_theory, timeout):
            return False

    return True

def smart_selection(
    candidates,
    base_theory,
    mode="filter",
    judge=None,
    occurrences=None,
    timeout=15,
    workers=4,
    group_size=5,
    max_rounds=3,
    verbose=False,
):
    """
    Strategy C:
      1. Filter consistency (parallel Prover9).
      2. Filter usefulness (parallel Prover9).
      3. If all compatible -> return ALL.
      4. If conflict -> LLM cascade -> return ONE.
    """
    if not candidates:
        return []

    # 1. Consistency
    consistent = filter_consistent_parallel(
        base_theory, candidates,
        timeout=timeout, workers=workers,
    )
    if not consistent:
        return []

    # 2. Usefulness
    useful = filter_useful_parallel(
        base_theory, consistent,
        timeout=timeout, workers=workers,
    )
    if not useful:
        return []

    # 3a. Trivial
    if len(useful) == 1:
        return useful

    # 3b. All compatible -> return all
    if are_all_compatible(useful, base_theory, timeout=timeout):
        if verbose:
            print(f"[strategy-C] {len(useful)} compatible axioms kept")
        return useful

    # 3c. Conflict -> LLM cascade
    if verbose:
        print(f"[strategy-C] conflict, LLM on {len(useful)} candidates")

    if judge is None:
        return [useful[0]]

    chosen = select_with_llm_cascade(
        useful, judge,
        mode=mode,
        occurrences=occurrences,
        group_size=group_size,
        max_rounds=max_rounds,
        verbose=verbose,
    )

    return [chosen] if chosen else []

def process_pair_candidate(
    cand,
    base_theory,
    judge=None,
    occurrences=None,
    timeout=15,
    workers=4,
    group_size=5,
    max_rounds=3,
    verbose=False,
):
    """
    Process one classified pair (Rule 1).

    Steps:
      1. Generate candidates from relation + arities.
      2. Apply Strategy C.

    Returns:
        list of axiom strings (possibly multiple if compatible)
    """
    left_name, left_arity = cand["left"]
    right_name, right_arity = cand["right"]
    relation = cand["relation"]

    candidates = build_candidates(
        left_name, left_arity,
        right_name, right_arity,
        relation,
    )
    if not candidates:
        return []

    return smart_selection(
        candidates,
        base_theory,
        judge=judge,
        occurrences=occurrences,
        timeout=timeout,
        workers=workers,
        group_size=group_size,
        max_rounds=max_rounds,
        verbose=verbose,
    )


# =========================================================
# Cluster processing (Rule 2)
# =========================================================
def process_cluster(
    cluster,
    base_theory,
    judge=None,
    occurrences=None,
    timeout=15,
    workers=4,
    group_size=5,
    max_rounds=3,
    max_subset_size=2,
    dag_has_binary=False,
    theta=0.8,
    verbose=False,
):
    """
    Process one cluster (Rule 2).

    Steps:
      1. Generate cluster candidates.
      2. Apply Strategy C.

    Returns:
        list of axiom strings
    """
    candidates = build_cluster_candidates(
        cluster,
        max_subset_size=max_subset_size,
        theta=theta,
        dag_has_binary=dag_has_binary,
    )
    if not candidates:
        return []

    return smart_selection(
        candidates,
        base_theory,
        mode="selection",
        judge=judge,
        occurrences=occurrences,
        timeout=timeout,
        workers=workers,
        group_size=group_size,
        max_rounds=max_rounds,
        verbose=verbose,
    )


# =========================================================
# Full processing
# =========================================================
def process_all_candidates(
    classified_pairs,
    base_theory,
    clusters=None,
    judge=None,
    profile_by_signature=None,
    timeout=15,
    workers=4,
    group_size=5,
    max_rounds=3,
    max_subset_size=2,
    dag_has_binary=False,
    theta=0.8,
    verbose=False,
):
    """
    Process all candidates (Rule 1 + Rule 2) with Strategy C.

    Args:
        classified_pairs: list from alpha.classify
        base_theory: string of FOL axioms
        clusters: list of clusters (Rule 2), each a list of dicts
        judge: optional LLMJudge
        profile_by_signature: dict {(name, arity): profile}
        timeout: Prover9 timeout
        workers: Prover9 threads
        group_size: LLM group size
        max_rounds: LLM cascade depth
        max_subset_size: Rule 2 subset size
        dag_has_binary: if True, use 'all x. all y.' prefix in clusters
        theta: Levenshtein threshold for cluster filter
        verbose: print progress

    Returns:
        dict with keys:
          - 'pair_axioms'    : list of unique axioms from Rule 1
          - 'cluster_axioms' : list of unique axioms from Rule 2
    """
    pair_axioms = []
    cluster_axioms = []

    # --- Rule 1: pairs ---
    for cand in classified_pairs:
        occ = []
        if profile_by_signature:
            left_name, left_arity = cand["left"]
            right_name, right_arity = cand["right"]
            p_left = profile_by_signature.get((left_name, left_arity))
            p_right = profile_by_signature.get((right_name, right_arity))
            if p_left:
                occ.extend(p_left.get("fol_occurrences", []))
            if p_right:
                occ.extend(p_right.get("fol_occurrences", []))

        axioms = process_pair_candidate(
            cand, base_theory,
            judge=judge,
            occurrences=occ,
            timeout=timeout,
            workers=workers,
            group_size=group_size,
            max_rounds=max_rounds,
            verbose=verbose,
        )
        pair_axioms.extend(axioms)

    pair_axioms = list(dict.fromkeys(pair_axioms))

    # --- Rule 2: clusters ---
    if clusters:
        for cluster in clusters:
            occ = []
            if profile_by_signature:
                for item in cluster:
                    profile = profile_by_signature.get(
                        (item["predicate"], item["arity"])
                    )
                    if profile:
                        occ.extend(profile.get("fol_occurrences", []))

            axioms = process_cluster(
                cluster, base_theory,
                judge=judge,
                occurrences=occ,
                timeout=timeout,
                workers=workers,
                group_size=group_size,
                max_rounds=max_rounds,
                max_subset_size=max_subset_size,
                dag_has_binary=dag_has_binary,
                theta=theta,
                verbose=verbose,
            )
            cluster_axioms.extend(axioms)

    cluster_axioms = list(dict.fromkeys(cluster_axioms))

    return {
        "pair_axioms": pair_axioms,
        "cluster_axioms": cluster_axioms,
    }
