"""
[WIP] Unfinished

Alternative script to use HF Transformers Pipeline for inference.
"""

# import datasets
import torch

from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
# from transformers.pipelines.pt_utils import KeyDataset

import deft


corpus_path = "data/test-medshake-score.json"
model_path = "meta-llama/Meta-Llama-3-8B"
prompt_template_id = "0"
num_shots = 2
shots_full_answer = False

TASK = "text-generation"
# TASK = "question-answering"


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
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# dataset = datasets.load_dataset(corpus_path)

# pipe = pipeline(model=model_path)
pipe = pipeline(task=TASK, model=model, tokenizer=tokenizer)

def generate(input_string):
    # outputs = model.generate(
    #     input_ids=inputs.input_ids.to("cuda"),
    #     attention_mask=inputs.attention_mask,
    #     max_new_tokens=32,
    #     pad_token_id=tokenizer.eos_token_id,

    #     # Handle optional parameters
    #     # https://huggingface.co/docs/transformers/en/generation_strategies
    #     **generate_kwargs
    # )
    generated = pipe(input_string)[0]['generated_text']
    return generated[len(input_string):]


results = deft.run_inference(
    generate, corpus_path, deft.template_from_id(prompt_template_id),
    num_shots=num_shots, include_full_answers=shots_full_answer,
)

print(results)
