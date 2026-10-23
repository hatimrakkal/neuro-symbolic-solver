"""Validation of claim-DAG responses.

A claim DAG is expected as a JSON object with a "label" and a list of
"steps"; each step must provide "claim_id", "claim" and "parents", with
unique IDs and known parent references.
"""

import json

def validate_dag(response):
    """Validate the structure of a JSON claim-DAG response.

    Raises ValueError/TypeError on malformed input; returns True otherwise.
    """
    try:
        dag = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("The response cannot be loaded as JSON")
    
    if not isinstance(dag, dict):
        raise TypeError("The response must be a dictionnary")

    required_for_dag = {"label", "steps"}
    missing = required_for_dag - set(dag)

    if missing:
        raise ValueError(f"The dag must contain {missing}")

    required_for_steps = {"claim_id", "claim", "parents"}
    ids = set()

    for index, node in enumerate(dag["steps"]):
        if not isinstance(node, dict):
            raise TypeError(f"Node {index} must be a dictionnary")
        
        missing = required_for_steps - set(node)
        
        if missing:
            raise ValueError(f"Node {index}: missing keys {missing}")
        
        claim_id = node["claim_id"]

        if claim_id in ids:
            raise ValueError(f"Duplicate claim id: {claim_id}")

        if not isinstance(node["claim"], str):
            raise TypeError(f"claim must be a string: {claim_id}")
        
        if not isinstance(node["parents"], list):
            raise TypeError(f"parents must be a list: {claim_id}")

        ids.add(claim_id)
    

    for node in dag["steps"]:
        unknown = set(node["parents"]) - ids

        if unknown:
            raise ValueError(f"Unknown dependencies in {node['claim_id']}: {unknown}")
    

    return True
