"""Natural-language to FOL translation with an instruction-tuned model.

Sentences are translated in batches through a chat template; outputs that
do not parse as first-order logic are retried up to `max_iter` times before
being reported as failures.
"""

import torch
import json
from .dag import validate_dag
from nltk.sem.logic import Expression



def validate_generated_fol(fol):
    """True if the generated string parses as a first-order logic expression."""
    try:
        expr = Expression.fromstring(fol)
        return True
    except Exception as e:
        return False


def translate(sentences, model, tokenizer, max_iter=2, batch_size=16):
    """Translate natural-language sentences into FOL formulas.

    Unparsable generations are retried conversationally up to `max_iter`
    times; sentences that still fail are returned as None.

    Args:
        sentences: list of natural-language sentences.
        model: Hugging Face causal language model.
        tokenizer: matching tokenizer (chat template required).
        max_iter: number of retry rounds for unparsable outputs.
        batch_size: generation batch size.

    Returns:
        list of FOL strings (or None for failed sentences), same order.
    """
    device = next(model.parameters()).device

    states = []
    for s in sentences:
        states.append({
            "sentence": s,
            "messages": [
                {"role": "system",
                 "content": "Translate natural-language statements into valid first-order logic. "
                            "Return only the formula."},
                {"role": "user", "content": s}
            ],
            "result": None,
            "error": None
        })

    for attempt in range(max_iter + 1):
        active_indices = [i for i, st in enumerate(states)
                          if st["result"] is None and st["error"] is None]
        if not active_indices:
            break

        for batch_start in range(0, len(active_indices), batch_size):
            batch_indices = active_indices[batch_start: batch_start + batch_size]

            formatted_batch = []
            for i in batch_indices:
                formatted = tokenizer.apply_chat_template(
                    states[i]["messages"],
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_batch.append(formatted)

            inputs = tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=128
                )

            prompt_len = inputs["input_ids"].shape[1]

            for j, i in enumerate(batch_indices):
                generated_ids = output_ids[j][prompt_len:]
                output = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                if validate_generated_fol(output):
                    states[i]["result"] = output
                    continue

                if attempt < max_iter:
                    states[i]["messages"].append({"role": "assistant", "content": output})
                    states[i]["messages"].append({
                        "role": "user",
                        "content": "The generated first-order logic is not in good format/parsable. Please regenerate."
                    })
                else:
                    states[i]["error"] = (
                        f"The generated FOL for the sentence '{states[i]['sentence']}' "
                        f"is not parsable after {max_iter} retries: {output}. "
                        f"To keep the execution, the response will be replaced by None"
                    )
                    states[i]["result"] = None

    for st in states:
        if st["error"] is not None:
            print(st["error"])

    return [st["result"] for st in states]


def translate_dag(response, model, tokenizer, max_iter=2, batch_size=16):
    """Translate every claim of a JSON claim-DAG into FOL.

    Validates the DAG structure, then translates each node's claim.

    Returns:
        list of nodes, each extended with "fol" and a "status" field.
    """
    validate_dag(response)
    dag = json.loads(response)["steps"]

    translated_dag = []
    sentences = [node["claim"] for node in dag]

    
    fols = translate(sentences, model, tokenizer, max_iter=max_iter, batch_size=batch_size)
        
    for node, fol in zip(dag, fols):
        translated_dag.append({
            **node,
            "fol": fol,
            "status": "success" if fol is not None else "failed"
        })


    return translated_dag
