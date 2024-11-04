"""
Script to generate MCQ responses using a given model checkpoint and test data.

Sampling options based on: https://huggingface.co/blog/how-to-generate
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Disable progress bars (cleaner logs)
import datasets
datasets.disable_progress_bar()


# HuggingFace authentication
from util.hugging_face import hf_login
hf_login()


def main(
    corpus_path: str,
    result_path: str,
    model_path: str,
    use_special_pad_token: bool = False,
    prompt_template_id: str = "0",
    num_shots: int = 0,
    shots_full_answer: bool = False,
    # Optional parameters to the model's "generate" method
    **generate_kwargs
):
    # Hyper-parameters
    generate_kwargs = {
        k: v
        for k, v in generate_kwargs.items()
        if v is not None
    }
    if len(generate_kwargs):
        print(
            "Hyperparameters:",
            *[f"  - {k}: {v}" for k, v in generate_kwargs.items()],
            "\n",
            sep="\n",
        )

    quant_config=BitsAndBytesConfig(
        load_in_8bit=True,
        # llm_int8_threshold=6.0,
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_type=torch.bfloat16,
        # llm_int8_enable_fp32_cpu_offload=True,
    )
    device_map = {
        "": 0
    }
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_special_pad_token:
        tokenizer.add_special_tokens(
            {"pad_token": "<|reserved_special_token_250|>"})
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,

            # Handle optional parameters
            # https://huggingface.co/docs/transformers/en/generation_strategies
            **generate_kwargs
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(
        generate, corpus_path, deft.template_from_id(prompt_template_id),
        num_shots=num_shots, include_full_answers=shots_full_answer,
    )
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
