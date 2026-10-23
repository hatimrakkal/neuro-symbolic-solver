"""Parallel (thread pool) Prover9 filters used by Strategy C.

Filters candidate axiom formulations by consistency against a base theory
and by usefulness (non-redundancy with what the theory already implies).
"""

from concurrent.futures import ThreadPoolExecutor

from .prover9 import is_theory_consistent, proves


def filter_consistent_parallel(
    base_theory,
    formulations,
    timeout=15,
    workers=8,
):
    """
    Parallel version of filter_consistent.

    Keeps only the formulations that do NOT make the theory inconsistent.

    Args:
        base_theory: string of FOL axioms (no trailing dot)
        formulations: list of candidate axiom strings
        timeout: Prover9 timeout per candidate (seconds)
        workers: number of threads

    Returns:
        list of consistent formulation strings (order preserved)
    """
    if not formulations:
        return []

    def _test(axiom):
        full_theory = base_theory + "\n" + axiom
        return axiom, is_theory_consistent(full_theory, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_test, formulations))

    return [ax for ax, ok in results if ok]


def filter_useful_parallel(
    base_theory,
    formulations,
    timeout=15,
    workers=8,
):
    """
    Parallel version of filter_useful.

    Keeps only the formulations that are NOT already implied by the theory.

    Args:
        base_theory: string of FOL axioms (no trailing dot)
        formulations: list of candidate axiom strings
        timeout: Prover9 timeout per candidate (seconds)
        workers: number of threads

    Returns:
        list of useful formulation strings (order preserved)
    """
    if not formulations:
        return []

    def _test(axiom):
        return axiom, not proves(base_theory, axiom, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_test, formulations))

    return [ax for ax, useful in results if useful]
