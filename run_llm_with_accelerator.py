"""
[WIP] Unfinished

Script to run inferences in parallel using HuggingFace Accelerator.

Based on:
https://medium.com/@mayvic/llm-multi-gpu-batch-inference-with-accelerate-edadbef3e239
https://github.com/huggingface/trl/blob/main/examples/scripts/ppo.py

Slurm:
https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh
"""

import json

import torch
from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader  #, Dataset, IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
)

import deft


INFERENCE_BATCH_SIZE = 8


# @dataclass
# class ScriptArguments:
#     model_path: str = field(
#         default="meta-llama/Meta-Llama-3-8B",
#         metadata={
#             "help": "Model to use, either a path on HuggingFace or a local"
#                     " folder.",
#         }
#     )
#     corpus_path: str = field(
#         metadata={
#             "help": "Path to the file with the corpus data used for inference.",
#         }
#     )
#     result_path: str = field(
#         metadata={
#             "help": "Path to the file where the results will be stored.",
#         }
#     )
#     prompt_template_id: str = field(
#         default="0",
#         metadata={
#             "help": "Id of the prompt template to be used.",
#         }
#     )
#     num_shots: int = field(
#         default=0,
#         metadata={
#             "help": "Number of few-shots to include in the prompt.",
#         }
#     )
#     shots_full_answer: bool = field(
#         default=False,
#         metadata={
#             "help": "Whether the few-shots include the full answer text.",
#         }
#     )


def get_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    tokenizer.padding_side = "left"
    return tokenizer


def get_model(model_path: str, accelerator: Accelerator):
    quant_config=BitsAndBytesConfig(
        load_in_8bit=True,
        # llm_int8_threshold=6.0,
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_type=torch.bfloat16,
        # llm_int8_enable_fp32_cpu_offload=True,
    )
    # device_map = {
    #     "": accelerator.local_process_index()
    # }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,  # Will automatically be set as True anyway
    )
    return model


def get_dataloader(corpus_path: str, prompt_args: dict, tokenizer) -> DataLoader:
    # ds = load_dataset("json", data_files=corpus_path)
    with open(corpus_path) as f:
        corpus = json.loads(f.read())
    ds = Dataset.from_list(corpus)

    # def tokenize(sample):
    #     tokens = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         # truncation=True, padding='max_length', max_length=256,
    #     )
    #     return tokens

    ds = ds.map(
        lambda s: {
            "prompt": deft.get_prompt(instance=s, few_shots_corpus=corpus,
                                      **prompt_args),
        },
        # batched=True,
        # batch_size=8,
        remove_columns=('question', 'answers', 'correct_answers',
                        'subject_name', 'nbr_correct_answers', 'medshake',
                        'medshake_difficulty'),
    )
    # tokenized.set_format(type="torch")
    # return tokenized
    dataloader = DataLoader(
        ds,
        # batch_size=INFERENCE_BATCH_SIZE,
        # collate_fn=DataCollatorWithPadding(tokenizer),
    )
    return dataloader


def run_generation(
        dataloader: DataLoader,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        accelerator: Accelerator,
):
    # model, dataloader = accelerator.prepare(model, dataloader)
    # model = accelerator.prepare(model)

    results = []

    for batch in tqdm(dataloader):
        with torch.inference_mode():
            # print(
            #     batch.keys(),                   # id, prompt
            #     len(batch["prompt"]),           # batch size
            # )
            inputs = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                # truncation=True,
            )
            # print(
            #     type(inputs["input_ids"]),      # tensor
            #     len(inputs["input_ids"]),       # batch size
            #     type(inputs["attention_mask"]), # tensor
            #     len(inputs["attention_mask"]),  # batch size
            # )
            # unwrapped_model = accelerator.unwrap_model(model)
            outputs = model.generate(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Needed so that the batch sizes will be the same across all workers,
            # as required by "gather_for_metrics"
            outputs = accelerator.pad_across_processes(
                outputs, dim=1, pad_index=tokenizer.pad_token_id)

            # Gathers the outputs from the workers into the main process
            outputs = accelerator.gather_for_metrics(outputs).cpu().tolist()

            # print(len(outputs))

            # generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # return generated[len(input_string):]

        outputs = [
            (id, tokenizer.decode(outs, skip_special_tokens=True))
            for id, outs in zip(batch["id"], outputs)
        ]
        results.extend(outputs)

    print(len(results))
    return results


def save_output(results: list[tuple[str, str]], result_path):
    def iter_lines():
        for id, generated in results:
            answer = deft.extract_answer(generated)
            answer = list(sorted(answer))
            yield id + ';' + '|'.join(answer)
    deft.write_results(iter_lines(), result_path)


def main(
    corpus_path: str,
    result_path: str,
    model_path: str = "meta-llama/Meta-Llama-3-8B",
    prompt_template_id: str = "0",
    num_shots: int = 0,
    shots_full_answer: bool = False,
):
    # Handle arguments
    # parser = HfArgumentParser(ScriptArguments)
    # args = parser.parse_args_into_dataclasses()

    # Init Accelerator
    accelerator = Accelerator()

    with accelerator.main_process_first():
        # HuggingFace authentication
        from util.hugging_face import hf_login
        hf_login()

        # Disable progress bars (cleaner logs)
        # import datasets
        # datasets.disable_progress_bar()

        tokenizer = get_tokenizer(model_path)
        model = get_model(model_path, accelerator)

        prompt_args = {
            "prompt_tpl": deft.template_from_id(prompt_template_id),
            "num_shots": num_shots,
            "include_full_answers": shots_full_answer,
        }
        dataloader = get_dataloader(corpus_path, prompt_args, tokenizer)

    model, dataloader = accelerator.prepare(model, dataloader)
    results = run_generation(dataloader, tokenizer, model, accelerator)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        save_output(results, result_path)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
