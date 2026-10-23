"""LLM-driven cascade selection among candidate axioms.

Candidates are split into groups, each group is scored by the LLM judge in
a single batched call, and surviving candidates are re-scored in rounds
until a single axiom remains. A Prover9-based "most general" fallback
resolves ties when the maximum number of rounds is reached.
"""

import re
from ..prover.prover9 import proves
from itertools import combinations

SELECTION_PROMPT = """You are an expert in first-order logic.

Given the FOL occurrences:
{occurrences}

Which of these candidate axioms is most likely to be correct?
{formulations}

Reply with ONLY the number of the best candidate (1-{n}), or "none" if none is correct.

Answer:"""


def _format_occurrences(occurrences, max_occ=8):
    """Render the FOL occurrences as a bulleted list (truncated)."""
    if not occurrences:
        return "(none)"
    return "\n".join(f"- {o}" for o in occurrences[:max_occ])


def _format_candidates(group):
    """Render a candidate group as a numbered list."""
    return "\n".join(f"{i+1}. {ax}" for i, ax in enumerate(group))


def build_selection_prompt(group, occurrences):
    """Build the selection prompt for one group of candidates."""
    return SELECTION_PROMPT.format(
        occurrences=_format_occurrences(occurrences),
        formulations=_format_candidates(group),
        n=len(group),
    )

def _parse_selection(response, n_candidates):
    """Extract the selected candidate index from an LLM response (or None)."""
    if not response:
        return None

    text = response.strip().lower()

    # Explicit "none"
    if "none" in text:
        return None

    # Find the first integer between 1 and n_candidates
    for match in re.finditer(r'\b([1-9]\d*)\b', text):
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx

    return None

def most_general_fols(list_of_fols, timeout=10):
    """Keep the pairwise most general formulas under Prover9 implication."""
    general = []
    
    for i, j in combinations(range(len(list_of_fols)), 2):

        fi = list_of_fols[i]
        fj = list_of_fols[j]

        fi_implies_fj = proves(fi, fj, timeout=timeout)
        fj_implies_fi = proves(fj, fi, timeout=timeout)

        if fi_implies_fj and not fj_implies_fi:
            general.append(fj)
            continue

        if fj_implies_fi and not fi_implies_fj:
            general.append(fi)
            continue
        
        general.extend([fi, fj])
    
    return general

def select_with_llm_cascade(
    candidates,
    judge,
    mode="selection",
    occurrences=None,
    group_size=5,
    max_rounds=3,
    temperature=0.0,
    verbose=False
):
    """
    Cascade selection with BATCHED LLM calls.

    Each round:
      1. Split candidates into groups of `group_size`.
      2. Build ALL prompts for the round.
      3. One batched LLM call.
      4. Parse each response and collect chosen.

    Rules:
      - 0 chosen across all groups -> return None
      - 1 chosen across all groups -> return it
      - >= 2 chosen -> recurse on the chosen subset

    Args:
        candidates: list of axiom strings (already filtered)
        judge: LLMJudge instance
        occurrences: list of FOL occurrences (context)
        group_size: number of candidates shown to the LLM at once
        max_rounds: maximum recursion depth
        temperature: LLM temperature
        verbose: print progress

    Returns:
        the chosen axiom string, or None
    """
    if not candidates:
        return None

    current = list(candidates)
    round_num = 0

    while current and round_num < max_rounds:

        # Trivial case: only one candidate left
        if len(current) == 1:
            return current[0]

        # --- Step 1: build groups ---
        groups = [
            current[start:start + group_size]
            for start in range(0, len(current), group_size)
        ]

        # --- Step 2: build all prompts for the round ---
        prompts = [
            build_selection_prompt(g, occurrences or [])
            for g in groups
        ]

        # --- Step 3: ONE batched LLM call ---
        responses = judge.decide(prompts, temperature=temperature)

        # --- Step 4: parse and collect ---
        chosen = []
        for group, response in zip(groups, responses):
            pick = _parse_selection(response, len(group))
            if pick is not None:
                chosen.append(group[pick])

        if verbose:
            print(f"[cascade] round {round_num+1}: "
                  f"{len(current)} candidates -> {len(chosen)} chosen "
                  f"({len(prompts)} prompts)")

        # --- Termination conditions ---
        if len(chosen) == 0:
            return None
        if len(chosen) == 1:
            return chosen[0]

        # --- Recurse ---
        current = chosen
        round_num += 1

    # Fallback: max rounds reached
    if not current:
        return None
    if verbose:
        print(f"[cascade] max rounds reached, fallback to first of {len(current)}")
    return most_general_fols(current)[0] if mode=="selection" else current[0]
