"""Local Hugging Face causal language model used as an LLM judge.

Batched, left-padded generation with chat templates and optional 4-bit
(NF4) quantization via bitsandbytes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMJudge:
    """Wrapper around a local causal LM that answers selection prompts in batches."""

    def __init__(self, model_id, batch_size=10,
                load_in_4bit=False, max_new_tokens=20):
       
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

   
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype="auto",
                device_map="auto",
            )

        self.model.eval()

    @torch.no_grad()
    def _generate(self, prompts, temperature=0.0,
                max_new_tokens=None, batch_size=None):
        """Generate one completion per prompt, processing prompts in batches."""

        if not prompts:
            return []

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if batch_size is None:
            batch_size = self.batch_size

        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if temperature <= 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[-1]
            for out in outputs:
                new_tokens = out[input_len:]
                all_outputs.append(
                    self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                )

        return all_outputs


    def decide(self, prompts, temperature=0.0, batch_size=None,
               max_new_tokens=None):
        """Answer a prompt or list of prompts; always returns a list of strings."""
    
        if isinstance(prompts, str):
            prompts = [prompts]

        return self._generate(
            prompts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
