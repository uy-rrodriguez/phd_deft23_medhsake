"""
Code for MCQ with BERT based on:
https://huggingface.co/docs/transformers/tasks/multiple_choice

Base BertForMultipleChoice described here:
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice
"""

import itertools
import json
import sys
from typing import Callable, Optional, Union

import datasets
# import evaluate
import numpy as np
import torch
from dataclasses import dataclass
# from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
# from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

import deft



# Disable progress bars (cleaner logs)
# datasets.disable_progress_bar()


# HuggingFace authentication
from util.hugging_face import hf_login
hf_login()


# Fixed number of choices to support batch processing
CHOICES = ["a", "b", "c", "d", "e"]


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
        labels = [instance.pop("label") for instance in instances]
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
    # choices = sorted(list(instance["answers"].keys()))
    new_answers = {}
    combinations = []
    for i in range(1, len(CHOICES) + 1):
        combinations.extend(itertools.combinations(CHOICES, i))
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


def load_dataset(
        train_corpus_path: str,
        dev_corpus_path: Optional[str] = None,
) -> datasets.DatasetDict:
    data_files = {
        "train": train_corpus_path,
        **({"validation": dev_corpus_path} if dev_corpus_path else {}),
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    # ds = ds.filter(lambda s: s["nbr_correct_answers"] == 1)
    # Add all possible combinations of answers as individual choices
    ds = ds.map(lambda s: {
        "original_answers": s["answers"],
        "answers": expand_answers(s),
    })
    # Define label as index of correct answer
    ds = ds.map(lambda s: {"label": get_correct_answer_idx(s)})
    return ds


def generate_instance_context(
        instance: dict, ctx_include_choices: bool = False, **kwargs
) -> str:
    """
    Given a sample from the dataset, returns the context created from the
    question and, optionally, all the answer choices.
    """
    ctx = instance["question"]
    if ctx_include_choices:
        ctx += "\n" + "\n".join([
            a
            for letter, a in instance["answers"].items()
            if letter in instance["original_answers"]
        ])
    return ctx


def generate_single_answer(
        answer: str, answer_add_prefix: bool = False, **kwargs
) -> str:
    """
    Given a single answer choice for a sample, returns the answer formatted
    appropriately, optionally including an introductory prefix.
    """
    if answer_add_prefix:
        return "Réponse(s) : " + answer
    return answer


def instance_tokenizer(
        batch: dict[str, list],
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
) -> dict[str, list]:
    """
    Tokenizes a list of instances as (question, answer) pairs.
    """
    num_choices = len(batch["answers"][0])

    # Make N copies of the question, N the number of choices.
    questions = [
        [generate_instance_context(
            {
                k: values[i]
                for i, k in enumerate(batch.keys())
            },
            **kwargs,
        )] * num_choices
        for values in zip(*[v for v in batch.values()])
    ]

    # Transform answers dictionary into arrays, each one choice for the
    # question
    answers = [
        [generate_single_answer(a, **kwargs) for a in a.values()]
        for a in batch["answers"]
    ]

    # Flatten the lists to tokenize them
    questions = sum(questions, [])
    answers = sum(answers, [])

    # Tokenize (question, answer) pairs
    # tokenized = tokenizer(questions, answers, truncation=True)
    tokenized = tokenizer(questions, answers, truncation=True, padding=True)

    # Unflatten them afterwards so each example has a corresponding
    # input_ids, attention_mask, and labels field.
    result = {
        k: [
            v[i : i + num_choices]
            for i in range(0, len(v), num_choices)
        ]
        for k, v in tokenized.items()
    }
    # Result is a dict like: {
    #   "input_ids": list[Tensor[1, 1]],
    #   "attention_mask": list[Tensor[1, 1]],
    # }
    return result


def preds_to_logits(predictions: list, fn: str = "sigmoid", threshold: int = 0.5):
    if fn == "sigmoid":
        fn = torch.nn.Sigmoid()
    elif fn == "relu":
        fn = torch.nn.ReLU()
    else:
        raise ValueError(f"Unrecognised function name '{fn}'")
    probs = fn(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred


def compute_metrics(pred: EvalPrediction):
    print("\nPrediction obj:", pred, file=sys.stderr)
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0]
    else:
        preds = pred.predictions

    y_pred = preds_to_logits(preds)
    # y_pred = preds_to_logits(predictions, fn="relu")
    print("\nLength:", len(preds), file=sys.stderr)
    print("Preds:\n", preds, file=sys.stderr)
    print("Any non-zero in Logits?:", np.any(y_pred), file=sys.stderr)
    print("Logits:\n", y_pred, file=sys.stderr)

    y_true = pred.label_ids
    print("Ref:\n", y_true, file=sys.stderr)

    metrics = {
        "f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "hamming": deft.batch_emr(y_pred.tolist(), y_true.tolist()),
        "emr": deft.batch_hamming(y_pred.tolist(), y_true.tolist()),
    }
    return metrics


def run_training(
    model_path: str,
    train_corpus_path: str,
    dev_corpus_path: str,
    train_run_name: str,
    new_model_path: str,
    train_output_dir: str = "train_results/",
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        clean_up_tokenization_spaces=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset and pre-process to tokenize items
    ds = load_dataset(train_corpus_path, dev_corpus_path)
    tokenized_ds = ds.map(
        instance_tokenizer,
        fn_kwargs={
            "tokenizer": tokenizer,
            **kwargs,
        },
        batched=True,
        batch_size=32,
    )

    # quant_config=BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     # llm_int8_threshold=6.0,
    #     # load_in_4bit=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_type=torch.bfloat16,
    #     # llm_int8_enable_fp32_cpu_offload=True,
    # )
    # device_map = {
    #     "": 0
    # }
    print(">>>>> BEFORE MODEL.from_pretrained", file=sys.stderr)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_path,
        # device_map=device_map,
        # torch_dtype=torch.float16,
        # quantization_config=quant_config,
    )

    # Train model
    training_args = TrainingArguments(
        run_name=train_run_name,
        output_dir=train_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",

        # From Yanis Labrak's DEFT 2023
        # learning_rate=2e-5,
        greater_is_better=True,
        metric_for_best_model="emr",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),

        # From Yanis Labrak's DEFT 2023
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Save trained model
    tokenizer.save_pretrained(new_model_path)
    trainer.model.save_pretrained(new_model_path)


def run_inference(
    model_path: str,
    corpus_path: str,
    result_path: str,
    debug: bool = False,
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # quant_config=BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     # llm_int8_threshold=6.0,
    #     # load_in_4bit=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_type=torch.bfloat16,
    #     # llm_int8_enable_fp32_cpu_offload=True,
    # )
    # device_map = {
    #     "": 0
    # }
    model = AutoModelForMultipleChoice.from_pretrained(
        model_path,
        # device_map=device_map,
        # torch_dtype=torch.float16,
        # quantization_config=quant_config,
    )

    # Load dataset and pre-process to tokenize items
    print("Loading dataset")
    ds = load_dataset(corpus_path)["train"]
    tokenized_ds = ds.map(
        instance_tokenizer,
        fn_kwargs={
            "tokenizer": tokenizer,
            **kwargs,
        },
        batched=True,
        batch_size=32,
    )

    results = []
    all_match = []
    all_hamming = []
    all_medshake = []

    # for i, instance in enumerate(tqdm(ds)):
    for i, instance in enumerate(tokenized_ds):
        print(i + 1, instance["question"])
        # print(json.dumps(instance["original_answers"], indent=2), file=sys.stderr)

        # Run the inference
        labels = torch.tensor(instance["label"]).unsqueeze(0)
        outputs = model(
            **{
                k: torch.tensor(instance[k]).unsqueeze(0)
                for k in ("input_ids", "attention_mask")
            },
            labels=labels,
        )
        logits: torch.Tensor = outputs.logits

        # Get the class with the highest probability
        predicted_class = logits.argmax().item()

        # Process answer
        answer_keys = sorted(instance["answers"].keys())
        answer = answer_keys[predicted_class].split(" ")
        print(answer, instance["correct_answers"], "Answer len:", len(answer))

        if debug:
            print(json.dumps(instance["answers"], indent=2))
            topk = logits.flatten().topk(len(instance["answers"]))  #3)
            soft = logits.flatten().softmax(0)
            print({
                answer_keys[i]: soft[i].item()
                for i in topk.indices
            })
            print("\n")

        is_exact_match = set(answer) == set(instance["correct_answers"])
        hamming_val = deft.hamming(answer, instance["correct_answers"])
        medshake = deft.medshake_rate(answer, instance["medshake"])

        results.append(instance["id"] + ";" + "|".join(answer))
        all_match.append(is_exact_match)
        all_hamming.append(hamming_val)
        all_medshake.append(medshake)

    print("EXACT MATCH:", np.average(all_match))
    print("HAMMING SCORE:", np.average(all_hamming))
    print("MEDSHAKE RATE:", np.average(all_medshake))
    deft.write_results(results, result_path)


def main(method_name: str, *args, **kwargs) -> None:
    import run_bert_mcq as module
    method = getattr(module, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
