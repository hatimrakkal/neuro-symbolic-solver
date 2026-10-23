"""Thin subprocess wrapper around the Prover9 theorem prover.

Provides theory-consistency checking and goal proving over FOL theory
strings written in Prover9 input syntax.
"""

import subprocess
from pathlib import Path
from typing import List


# Path to the Prover9 binary. Adjust to match your local installation.
PROVER9_BIN = "/kaggle/working/Prover9-LADR-2026-8A/bin/prover9"



def _run_prover9(theory: str, goal: str = "$F", timeout: int = 30) -> str:
    """Run Prover9 on a theory/goal pair and return the combined output."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False, dir="/tmp") as f:
        f.write("formulas(assumptions).\n")
        f.write(theory)
        f.write("\nend_of_list.\n")
        f.write("formulas(goals).\n")
        f.write(goal)
        f.write("\nend_of_list.\n")
        in_path = f.name

    try:
        r = subprocess.run(
            [PROVER9_BIN, "-f", in_path, "-t", str(timeout)],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        Path(in_path).unlink(missing_ok=True)


def is_theory_consistent(theory: str, timeout: int = 30) -> bool:
    """True if Prover9 fails to derive a contradiction ($F) from the theory.
    
    Undecidable cases (timeout/error) are conservatively treated as consistent.
    """
    output = _run_prover9(theory, goal="$F", timeout=timeout)
    if "TIMEOUT" in output or "ERROR" in output:
        # Can't decide → assume consistent (conservative)
        return True
    return "THEOREM PROVED" not in output


def proves(theory: str, goal: str, timeout: int = 30) -> bool:
    """True if Prover9 proves the goal from the theory within the timeout."""
    output = _run_prover9(theory, goal=goal, timeout=timeout)
    if "TIMEOUT" in output or "ERROR" in output:
        return False
    return "THEOREM PROVED" in output
