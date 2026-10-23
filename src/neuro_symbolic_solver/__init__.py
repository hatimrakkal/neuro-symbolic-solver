"""Neuro-symbolic solver: alignment of first-order logic theories.

Research pipeline that turns natural-language claim DAGs into
first-order logic, then aligns the resulting theories by building
"bridge axioms":

- ``lexicon``  — tokenization, stemming and lexical similarity;
- ``fol``      — DAG validation, NL-to-FOL translation, predicate/constant
  extraction, profiling and formula normalization;
- ``axioms``   — candidate axiom generation, cluster expansion, Strategy C
  processing and bridge-axiom construction;
- ``llm``      — local LLM judge and cascade selection;
- ``prover``   — Prover9 wrapper and parallel consistency/usefulness filters.

Submodules are imported lazily by the caller, e.g.::

    from neuro_symbolic_solver.axioms.construction import build_bridge_axioms
"""
