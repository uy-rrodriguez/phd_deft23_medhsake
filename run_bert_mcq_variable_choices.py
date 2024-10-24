"""
This more complex implementation of run_bert_mcq.py attempts to support a varied
number of choices per question. Tokens are generated for the pairs
question-answer, for each question and each answer as normal. Then, additional
pairs of zeros are added so each instance has the same number of total choices.

The collator for batch processing is the same, since at this point all instances
have the same number of choices.
"""

import itertools
import sys
from typing import Callable, Optional, Union

import datasets
# import evaluate
import numpy as np
import torch
from dataclasses import dataclass
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch, LazyRow
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)



# Disable progress bars (cleaner logs)
# datasets.disable_progress_bar()


# HuggingFace authentication
from util.hugging_face import hf_login
hf_login()


# Fixed number of choices to support batch processing
# CHOICES = ["a", "b", "c", "d", "e"]


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice
    received.

    Transformers doesn't have a data collator for multiple choice, so we need to
    adapt the "DataCollatorWithPadding" to create a batch of examples. It's more
    efficient to dynamically pad the sentences to the longest length in a batch
    during collation, instead of padding the whole dataset to the maximum
    length.

    "DataCollatorForMultipleChoice" flattens all the model inputs, applies
    padding, and then unflattens the results.

    This implementation requires a consistent number of choices per batch.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: list):
        print(">>>>> DataCollatorForMultipleChoice.call", file=sys.stderr)
        # print(type(instances[0]), file=sys.stderr)
        # print(instances[0].keys(), file=sys.stderr)
        # print(instances[0], file=sys.stderr)
        # exit

        label_name = "label" if "label" in instances[0].keys() else "labels"
        labels = [instance.pop(label_name) for instance in instances]
        batch_size = len(instances)
        num_choices = len(instances[0]["input_ids"])
        flattened_instances = [
            [
                {k: v[i] for k, v in instance.items()}
                for i in range(num_choices)
            ]
            for instance in instances
        ]
        flattened_instances = sum(flattened_instances, [])

        batch = self.tokenizer.pad(
            flattened_instances,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Unflatten items in batch
        batch = {
            k: v.view(batch_size, num_choices, -1)
            for k, v in batch.items()
        }
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def expand_answers(instance: dict) -> dict[str, str]:
    choices = sorted(list(instance["answers"].keys()))
    new_answers = {}
    combinations = []
    for i in range(1, len(choices) + 1):
        combinations.extend(itertools.combinations(choices, i))
    for comb in combinations:
        comb = sorted(comb)
        new_answers[" ".join(comb)] = "; ".join([
            f"({letter}) {instance['answers'][letter]}"
            for letter in comb
        ])
    return new_answers


def get_correct_answer_idx(instance: dict) -> int:
    choices = sorted(instance["answers"].keys())
    idx = choices.index(" ".join(sorted(instance["correct_answers"])))
    return idx


def load_dataset(train_corpus_path: str, dev_corpus_path: str) -> Dataset:
    data_files = {
        "train": train_corpus_path,
        "validation": dev_corpus_path,
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    # TODO: Remove filtering of questions with only one correct answer
    ds = ds.filter(lambda s: s["nbr_correct_answers"] == 1)
    # Add all possible combinations of answers as individual choices
    # ds = ds.map(lambda s: {"answers": expand_answers(s)})
    # Define label as index of correct answer
    ds = ds.map(lambda s: {"label": get_correct_answer_idx(s)})
    return ds


def build_instance_tokenizer(tokenizer: PreTrainedTokenizerBase) \
        -> Callable[[list], str]:
    """
    Returns a function that will tokenize a list of instances as (question,
    answer) pairs.
    """
    def func(instances: LazyBatch) -> dict[str, list]:
        print(">>>>> instance_tokenizer", file=sys.stderr)

        # Make N copies of the question, N the number of choices.
        # If questions have different numbers of choices, some items in answers
        # will be null and we have to handle them properly, to make every
        # instance have the same number of choices.
        questions = [
            [q for v in a.values() if v]  # Exclude null answers
            for q, a in zip(instances["question"], instances["answers"])
        ]
        # Transform answers dictionary into arrays, each one choice for the
        # question
        answers = [
            [v for v in a.values() if v]  # Exclude null answers
            for a in instances["answers"]
        ]
        # print(questions, file=sys.stderr)
        # print(answers, file=sys.stderr)

        # Flatten the lists to tokenize them
        questions = sum(questions, [])
        answers = sum(answers, [])

        # Tokenize (question, answer) pairs
        tokenized = tokenizer(questions, answers, truncation=True)

        # Unflatten them afterwards so each example has a corresponding
        #  input_ids, attention_mask, and labels field.
        result = {}
        for k, tokens_list in tokenized.items():
            iter_t = iter(tokens_list)
            result[k] = [
                [
                    next(iter_t) if a else [0]  # Replace tokens by 0s for null answers
                    for a in answers.values()
                ]
                for answers in instances["answers"]
            ]
        # Result is a dict like: {
        #   "input_ids": list[Tensor[1, 1]],
        #   "attention_mask": list[Tensor[1, 1]],
        # }
        # print(result, file=sys.stderr)
        # raise RuntimeError()
        return result
    return func


def train(
        model: AutoModelForMultipleChoice, tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset, run_name: str, output_dir: str,
) -> None:
    # Include a metric during training for evaluating the model’s performance
    # accuracy = evaluate.load("accuracy")
    # def compute_metrics(eval_pred: tuple) -> dict:
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        # compute_metrics=compute_metrics,
    )
    print(">>>>> BEFORE TRAIN", file=sys.stderr)
    trainer.train()


def run_training(
    model_path: str = "Dr-BERT/DrBERT-4GB",
    train_corpus_path: str = "data/train_variable_choices.json",
    dev_corpus_path: str = "data/dev.json",
    train_run_name: str = "drbert-4gb-deft_001",
    train_out_path: str = "train_results/drbert/001_20240904",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        clean_up_tokenization_spaces=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    print(">>>>> BEFORE MODEL.from_pretrained", file=sys.stderr)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_path,
        # device_map=device_map,
        # torch_dtype=torch.float16,
        # quantization_config=quant_config,

        # # DrBERT
        # # Remove warning about using "CamembertLMHeadModel" as a standalone
        # is_decoder=True,
    )

    # Load dataset and pre-process to tokenize items
    print(">>>>> BEFORE LOAD DATASET", file=sys.stderr)
    ds = load_dataset(train_corpus_path, dev_corpus_path)
    print(">>>>> BEFORE DATASET MAP", file=sys.stderr)
    tokenized_ds = ds.map(
        build_instance_tokenizer(tokenizer),
        batched=True,
        batch_size=16,
    )

    # Train model
    train(model, tokenizer, tokenized_ds, train_run_name, train_out_path)


def run_inference(
    model_path: str = "train_results/drbert/001_20240827",
    corpus_path: str = "data/dev.json",
    result_path: str = "output/drbert/tuned_001_20240827",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    model = AutoModelForMultipleChoice.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quant_config,

        # # DrBERT
        # # Remove warning about using "CamembertLMHeadModel" as a standalone
        # is_decoder=True,
    )

    # Load dataset and pre-process to tokenize items
    ds = load_dataset(corpus_path)
    tokenized_ds = ds.map(
        build_instance_tokenizer(tokenizer),
        batched=True,
        batch_size=2,
    )
    labels = torch.tensor(0).unsqueeze(0)

    # Run the inference
    outputs = model.generate(
        **{
            k: v.unsqueeze(0)
            for k, v in tokenized_ds.items()
        },
        labels=labels,
    )
    logits = outputs.logits
    print(logits)

    # Get the class with the highest probability
    predicted_class = logits.argmax().item()
    print(predicted_class)


def main(method_name: str, *args, **kwargs) -> None:
    import run_bert_mcq_variable_choices as module
    method = getattr(module, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
