# neuro-symbolic-solver

Neuro-symbolic pipeline for **aligning first-order logic (FOL) theories**.

The project takes a set of claim DAGs (natural-language claims organized as
a directed acyclic graph), translates each claim into a first-order logic
formula with a local instruction-tuned language model, and then builds
**bridge axioms** that link the predicates and constants of the resulting
theories. Candidate axioms are generated combinatorially, filtered by the
**Prover9** theorem prover (consistency + usefulness, in parallel), and
conflicts are resolved by an **LLM judge** through a batched cascade
selection.

The NL2FOL translation is performed with **`teaislife/Qwen3-4B-nl2fol`**,
a 4B instruction-tuned model fine-tuned to emit Prover9 syntax.

## How it works

```
 claim DAG (JSON)
       |
       v
 [fol.nl2fol]        translate each claim into FOL (local LLM: teaislife/Qwen3-4B-nl2fol, batched, retries)
       |
       v
 [fol.normalization] decompose complex atoms / Skolemize existential binders
       |
       v
 [fol.extraction]    extract predicate signatures and constants, build profiles
       |             and co-occurrence "cannot links"
       v
 [lexicon.alpha]     classify predicate pairs (EQUIVALENT, NEGATION,
       |             LEFT/RIGHT_CONTAINS, PROJECTION, UNRELATED)
       v
 [axioms.candidates] enumerate candidate bridge axioms (Rule 1: pairs,
       |             Rule 2: clusters)
       v
 [prover.prover9]    parallel consistency + usefulness filtering (Strategy C)
       |
       v
 [llm]               LLM cascade selection on conflicts (batched prompts)
       |
       v
 bridge axioms (+ equality axioms for aligned constants)
```

## Repository layout

```
neuro-symbolic-solver/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── src/
    └── neuro_symbolic_solver/
        ├── lexicon/            # lexical matching layer
        │   ├── similarity.py   #   tokenization, stemming, Levenshtein similarities
        │   ├── alpha.py        #   predicate-pair relation classification
        │   └── constants.py    #   constant-pair classification (SAME_ENTITY/UNRELATED)
        ├── fol/                # first-order logic layer
        │   ├── dag.py          #   claim-DAG JSON validation
        │   ├── nl2fol.py       #   NL -> FOL translation with a local LLM (teaislife/Qwen3-4B-nl2fol)
        │   ├── extraction.py   #   predicate/constant extraction, cannot links
        │   ├── profiles.py     #   per-signature and per-constant profiles
        │   └── normalization_new.py  # atom decomposition + Skolemization
        ├── axioms/             # bridge-axiom layer
        │   ├── candidates.py   #   candidate enumeration (pairs and clusters)
        │   ├── cluster_expansion.py  # sub-predicate cluster expansion
        │   ├── pipeline.py     #   Strategy C: Prover9 filters + LLM cascade
        │   └── construction.py #   orchestrator (build_bridge_axioms)
        ├── llm/                # LLM layer
        │   ├── llm_judge.py    #   local HF causal LM judge (batched generation)
        │   └── llm_selector.py #   cascade selection with Prover9 fallback
        └── prover/             # theorem-proving layer
            ├── prover9.py          # subprocess wrapper around Prover9
            └── prover9_parallel.py # thread-pool consistency/usefulness filters
```

## Installation

Requires **Python 3.9+** and a local [Prover9](https://www.cs.unm.edu/~mccune/prover9/)
installation.

```bash
git clone https://github.com/<your-account>/neuro-symbolic-solver.git
cd neuro-symbolic-solver

python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt

# NLTK data required by the lexical layer
python -c "import nltk; nltk.download('wordnet')"
```

### Prover9 setup

The binary path is defined as a module constant:

```python
# src/neuro_symbolic_solver/prover/prover9.py
PROVER9_BIN = "/kaggle/working/Prover9-LADR-2026-8A/bin/prover9"
```

Adjust `PROVER9_BIN` to the location of your local `prover9` executable.
Note that `metrics`-style evaluation is out of scope for this repository;
only the alignment pipeline is included.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuro_symbolic_solver.llm.llm_judge import LLMJudge
from neuro_symbolic_solver.fol.nl2fol import translate_dag
from neuro_symbolic_solver.axioms.construction import build_bridge_axioms

# 1. Load the translation model (teaislife/Qwen3-4B-nl2fol, emits Prover9 syntax)
#    and the judge model
model_id = "teaislife/Qwen3-4B-nl2fol"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
judge = LLMJudge(model_id, load_in_4bit=True)

# 2. A claim DAG as a JSON string:
#    {"label": "...", "steps": [{"claim_id": "c1", "claim": "...", "parents": []}, ...]}
dag_json = ...

# 3. Translate the claims into FOL
translated = translate_dag(dag_json, model, tokenizer, max_iter=2, batch_size=16)

# 4. Build the bridge axioms (judge=None -> deterministic fallback, no LLM)
axioms_by_dag = build_bridge_axioms(
    [translated],
    judge=judge,
    theta=0.8,
    base_theory="",
    timeout=15,
    workers=4,
    verbose=True,
)

print(axioms_by_dag[0])
```

## Main parameters

| Parameter | Default | Role |
|---|---|---|
| `theta` | `0.8` | lexical (token containment / name similarity) threshold for pair and cluster classification |
| `constant_similarity_threshold` | `0.85` | normalized Levenshtein threshold marking a constant pair as `SAME_ENTITY` |
| `timeout` | `15` | Prover9 timeout per call (seconds) |
| `workers` | `4` | thread-pool size for parallel Prover9 calls |
| `group_size` | `5` | number of candidates shown to the LLM judge per prompt |
| `max_rounds` | `3` | maximum depth of the LLM cascade |
| `max_subset_size` | `2` | maximum left/right subset size for cluster axioms (Rule 2) |
| `max_cluster_size` | `6` | cap on cluster size after sub-predicate expansion |

## Notes

- The NL2FOL translation uses **`teaislife/Qwen3-4B-nl2fol`**, a 4B
  instruction-tuned model fine-tuned to emit Prover9 syntax.
- The LLM layers expect any Hugging Face causal LM with a chat template;
  4-bit quantization (NF4) requires `bitsandbytes`
  (`pip install .[quantization]`).
- `device_map="auto"` requires `accelerate` (installed by default with this
  package).
- This is research code: the Prover9 path is a plain constant and failure
  handling is intentionally conservative (undecidable = consistent).
