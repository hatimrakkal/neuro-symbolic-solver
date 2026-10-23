"""Normalization of generated FOL formulas.

Two lexical, structure-preserving transforms are applied:

1. Decomposition: complex unary atoms whose predicate name covers known
   constants are rewritten as smaller predicates over those constants.
2. Skolemization: existential variables bound by a single predicate whose
   name lexically matches a known constant are replaced by that constant.

Thresholds control how strict the token-coverage matching is.
"""

from nltk.sem.logic import Expression
from nltk.sem.logic import AllExpression, ExistsExpression, LambdaExpression
from nltk.sem.logic import ApplicationExpression, BinaryExpression, NegatedExpression
from nltk.sem.logic import ConstantExpression, IndividualVariableExpression

from .extraction import extract_constants
import re
from rapidfuzz.distance import Levenshtein

def tokenize_predicate(name):
    """Split a CamelCase name into lowercase tokens."""
    tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
    return [t.lower() for t in tokens]

def token_similarity(t1, t2):
    """Normalized Levenshtein similarity between two tokens (case-insensitive)."""
    return Levenshtein.normalized_similarity(t1.lower(), t2.lower())

def covered_tokens(source_tokens, target_tokens, threshold):
    """Match every source token to its best target position above the threshold.

    Returns a list of (position, similarity) pairs, or None if any source
    token fails to reach the threshold.
    """
    positions = []

    for source_token in source_tokens:
        best_position = None
        best_similarity = 0.0

        for position, target_token in enumerate(target_tokens):
            similarity = token_similarity(source_token, target_token)
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = position

        if best_similarity <= threshold:
            return None

        positions.append((best_position, best_similarity))

    return positions


def skolem_constant(predicate, constants, threshold):
    """Find the constant best covered by a predicate name.

    Returns (best_constant, score); (None, 0.0) when nothing matches.
    """
    predicate_tokens = tokenize_predicate(predicate)
    best_constant = None
    best_score = 0.0

    for constant in constants:
        constant_tokens = tokenize_predicate(constant)

        if not constant_tokens:
            continue

        positions = covered_tokens(constant_tokens, predicate_tokens, threshold)

        if positions is None:
            continue

        score = sum(similarity for _, similarity in positions) / len(positions)

        if score > best_score:
            best_score = score
            best_constant = constant

    return best_constant, best_score


def collect_atoms(node, bound, atoms):
    """Recursively collect (predicate, argument-status list) atoms of a formula.

    Arguments bound by the leading existential chain are recorded with their
    variable name; all other arguments are recorded as None.
    """
    if isinstance(node, (AllExpression, ExistsExpression, LambdaExpression)):
        collect_atoms(node.term, {**bound, str(node.variable): "inner"}, atoms)
        return

    if isinstance(node, BinaryExpression):
        collect_atoms(node.first, bound, atoms)
        collect_atoms(node.second, bound, atoms)
        return

    if isinstance(node, NegatedExpression):
        collect_atoms(node.term, bound, atoms)
        return

    if isinstance(node, ApplicationExpression):
        try:
            name = node.pred.variable.name
        except AttributeError:
            return

        arguments = []
        for argument in node.args:
            if isinstance(argument, IndividualVariableExpression):
                variable = str(argument.variable)
                arguments.append(variable if bound.get(variable) == "leading" else None)
            else:
                arguments.append(None)

        atoms.append((name, arguments))

        for argument in node.args:
            collect_atoms(argument, bound, atoms)


def plan_skolemization(expr, constants, threshold):
    """Plan the Skolemization of a leading existential chain.

    For each existentially quantified variable bound by exactly one atom,
    look for a constant lexically covered by that predicate's name.

    Returns:
        (mapping: variable -> constant name, removals: (binder, variable) pairs)
    """
    chain = []

    while isinstance(expr, ExistsExpression):
        chain.append(str(expr.variable))
        expr = expr.term

    atoms = []
    collect_atoms(expr, {variable: "leading" for variable in chain}, atoms)

    mapping = {}
    removals = []
    binders = {}

    for variable in chain:
        best_binder = None
        best_score = 0.0

        for name, arguments in atoms:
            if arguments != [variable]:
                continue

            constant, score = skolem_constant(name, constants, threshold)

            if constant is not None and score > best_score:
                best_score = score
                best_binder = name

        if best_binder is None:
            continue

        binders.setdefault(best_binder, []).append(variable)
        removals.append((best_binder, variable))

    for binder, variables in binders.items():
        base = binder.lower()

        if len(variables) == 1:
            mapping[variables[0]] = base
        else:
            for index, variable in enumerate(variables, start=1):
                mapping[variable] = f"{base}{index}"

    return mapping, removals


def render_argument(argument, mapping, removals, bound):
    """Render one atom argument, isolating applied predicates from removals."""
    if isinstance(argument, ApplicationExpression):
        return rebuild_skolemization(argument, mapping, frozenset(), bound)

    return rebuild_skolemization(argument, mapping, removals, bound)


def rebuild_skolemization(node, mapping, removals, bound):
    """Rebuild a formula string with the planned Skolemization applied.

    Bound variables marked "kept" are preserved; dropped existentials are
    replaced by their constant, and atoms whose only argument was removed
    disappear (returning None so the parent can simplify).
    """
    if isinstance(node, IndividualVariableExpression):
        variable = str(node.variable)
        if variable in mapping and bound.get(variable) != "kept":
            return mapping[variable]
        return variable

    if isinstance(node, (AllExpression, ExistsExpression, LambdaExpression)):
        variable = str(node.variable)
        quantifier = "all"

        if isinstance(node, ExistsExpression):
            quantifier = "exists"
            if variable in mapping and variable not in bound:
                return rebuild_skolemization(node.term, mapping, removals, {**bound, variable: "dropped"})
        elif isinstance(node, LambdaExpression):
            quantifier = "\\"

        term = rebuild_skolemization(node.term, mapping, removals, {**bound, variable: "kept"})

        if term is None:
            return None

        return f"{quantifier} {variable}. {term}"

    if isinstance(node, BinaryExpression):
        operator = node.getOp()

        if operator == "&":
            left = rebuild_skolemization(node.first, mapping, removals, bound)
            right = rebuild_skolemization(node.second, mapping, removals, bound)

            if left is None:
                return right
            if right is None:
                return left

            return f"({left} & {right})"

        if operator == "->":
            left = rebuild_skolemization(node.first, mapping, removals, bound)
            right = rebuild_skolemization(node.second, mapping, removals, bound)

            if left is None:
                return right

            return f"({left} -> {right})"

        left = rebuild_skolemization(node.first, mapping, removals, bound)
        right = rebuild_skolemization(node.second, mapping, removals, bound)

        if left is None:
            left = rebuild_skolemization(node.first, mapping, frozenset(), bound)
        if right is None:
            right = rebuild_skolemization(node.second, mapping, frozenset(), bound)

        return f"({left} {operator} {right})"

    if isinstance(node, NegatedExpression):
        term = rebuild_skolemization(node.term, mapping, removals, bound)

        if term is None:
            term = rebuild_skolemization(node.term, mapping, frozenset(), bound)

        return f"-{term}"

    if isinstance(node, ApplicationExpression):
        try:
            name = node.pred.variable.name
        except AttributeError:
            return str(node)

        arguments = node.args

        if len(arguments) == 1 and isinstance(arguments[0], IndividualVariableExpression):
            variable = str(arguments[0].variable)
            if (name, variable) in removals and bound.get(variable) != "kept":
                return None

        rendered = [render_argument(argument, mapping, removals, bound) for argument in arguments]
        return f"{name}({', '.join(rendered)})"

    return str(node)


def apply_skolemization(expr, mapping, removals):
    """Apply a planned Skolemization; returns None when nothing has to change."""
    if not mapping:
        return None

    result = rebuild_skolemization(expr, mapping, set(removals), {})

    if result is None:
        binder, variable = removals[0]
        return f"{binder}({mapping[variable]})"

    return result


def decomposition_candidates(predicate, constants, threshold):
    """Enumerate constants that could decompose a unary predicate atom.

    A constant qualifies when its tokens are covered by the predicate's
    tokens and leave at least one token unused.
    """
    predicate_tokens = tokenize_predicate(predicate)
    candidates = []

    for constant in constants:
        constant_tokens = tokenize_predicate(constant)

        if not constant_tokens or len(constant_tokens) >= len(predicate_tokens):
            continue

        positions = covered_tokens(constant_tokens, predicate_tokens, threshold)

        if positions is None:
            continue

        matched = {position for position, _ in positions}

        if len(matched) < len(constant_tokens):
            continue

        candidates.append({
            "constant": constant,
            "position": min(matched),
            "score": sum(similarity for _, similarity in positions) / len(positions),
            "matched": matched,
        })

    return candidates


def decompose_atom(predicate, variable, constants, threshold):
    """Rewrite a unary atom as a new predicate over constants and the variable.

    Selects compatible constant candidates, builds the CamelCase predicate
    name from the remaining tokens, and places constants plus the variable
    as arguments. Returns None when no decomposition is possible.
    """
    candidates = decomposition_candidates(predicate, constants, threshold)

    if not candidates:
        return None

    selected = []
    used_positions = set()
    used_indices = set()
    upper = len(candidates) + 1

    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["constant"])):
        position = min(candidate["position"] + 1, upper)

        if position in used_positions:
            continue

        merged = used_indices | candidate["matched"]

        if len(merged) > len(tokenize_predicate(predicate)) - 1:
            continue

        selected.append((position, candidate))
        used_positions.add(position)
        used_indices = merged

    while True:
        total = len(selected) + 1
        filtered = [item for item in selected if item[0] <= total]

        if len(filtered) == len(selected):
            break

        selected = filtered

    if not selected:
        return None

    used_indices = set()
    for _, candidate in selected:
        used_indices |= candidate["matched"]

    base_tokens = [token for index, token in enumerate(tokenize_predicate(predicate)) if index not in used_indices]
    base = "".join(token.capitalize() for token in base_tokens)

    if not base:
        return None

    selected.sort(key=lambda item: item[0])
    total = len(selected) + 1
    constants_by_position = {position: candidate["constant"] for position, candidate in selected}
    variable_position = next(index for index in range(1, total + 1) if index not in constants_by_position)

    arguments = []
    for index in range(1, total + 1):
        if index == variable_position:
            arguments.append(variable)
        else:
            arguments.append(constants_by_position[index])

    return f"{base}({', '.join(arguments)})"


def rebuild_decomposition(node, constants, threshold):
    """Rebuild a formula string, decomposing unary atoms where possible."""
    if isinstance(node, (AllExpression, ExistsExpression, LambdaExpression)):
        variable = str(node.variable)
        quantifier = "all"

        if isinstance(node, ExistsExpression):
            quantifier = "exists"
        elif isinstance(node, LambdaExpression):
            quantifier = "\\"

        term = rebuild_decomposition(node.term, constants, threshold)
        return f"{quantifier} {variable}. {term}"

    if isinstance(node, BinaryExpression):
        left = rebuild_decomposition(node.first, constants, threshold)
        right = rebuild_decomposition(node.second, constants, threshold)
        return f"({left} {node.getOp()} {right})"

    if isinstance(node, NegatedExpression):
        return f"-{rebuild_decomposition(node.term, constants, threshold)}"

    if isinstance(node, ApplicationExpression):
        try:
            name = node.pred.variable.name
        except AttributeError:
            return str(node)

        rendered = [str(argument) for argument in node.args]

        if len(node.args) == 1 and isinstance(node.args[0], IndividualVariableExpression):
            replaced = decompose_atom(name, rendered[0], constants, threshold)
            if replaced is not None:
                return replaced

        return f"{name}({', '.join(rendered)})"

    return str(node)


def normalize_fol(fol, constants, skolem_threshold=0.9, decomposition_threshold=0.8):
    """Normalize one FOL string (decomposition, then Skolemization).

    Unparsable input is returned unchanged; each transform is skipped when
    its result does not re-parse.
    """
    try:
        expr = Expression.fromstring(fol)
    except Exception as error:
        print(f"Error : {error}")
        return fol

    result = rebuild_decomposition(expr, constants, decomposition_threshold)

    try:
        expr = Expression.fromstring(result)
    except Exception as error:
        print(f"Error : {error}")
        return result

    mapping, removals = plan_skolemization(expr, constants, skolem_threshold)
    skolemized = apply_skolemization(expr, mapping, removals)

    if skolemized is not None:
        result = skolemized

    return result


def process_translated_dag(translated_dag, skolem_threshold=0.9, decomposition_threshold=0.8):
    """Normalize the FOL of every node of a translated DAG.

    Keeps the original formula under "fol_original" and stores the
    normalized one under "fol".
    """
    fols = [node.get("fol") for node in translated_dag if node.get("fol")]
    constants = extract_constants(fols)

    normalized_dag = []

    for node in translated_dag:
        fol = node.get("fol")

        if not fol:
            normalized_dag.append(node)
            continue

        new_fol = normalize_fol(fol, constants, skolem_threshold, decomposition_threshold)
        normalized_dag.append({
            **node,
            "fol_original": fol,
            "fol": new_fol
        })

    return normalized_dag
