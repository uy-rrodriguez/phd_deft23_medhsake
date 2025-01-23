"""
Based on Yanis Labrak's code for DEFT 2023.

Test code to try answering multiple choice questions using
BertForSequenceClassification.
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification

As described in this answer https://stackoverflow.com/a/60615939 using a model
meant for sequence might not be ideal in our scenario, where the "classes"
(choices) change from sample to sample.
"""

import sys
from typing import Optional


# Wandb authentication, used to track training
# (Done at the very beginning, otherwise there is an error with PyArrow,
# imported by other packages)
from util.w_and_b import wandb_login


import datasets
# import evaluate
import numpy as np
import torch
# from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

import deft



# Disable progress bars (cleaner logs)
datasets.disable_progress_bar()


# HuggingFace authentication
from util.hugging_face import hf_login
hf_login()


# f1_metric = evaluate.load("f1")
# accuracy_metric = evaluate.load("accuracy")


# Fixed number of choices to support batch processing
CHOICES = ["a", "b", "c", "d", "e"]
id2label = {idx: label for idx, label in enumerate(CHOICES)}

BATCH_SIZE = 32


def load_dataset(
        train_corpus_path: str,
        dev_corpus_path: Optional[str] = None,
) -> datasets.DatasetDict:
    data_files = {
        "train": train_corpus_path,
        **({"validation": dev_corpus_path} if dev_corpus_path else {}),
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    # Initialise "labels" field for multi-label classification
    ds = ds.map(
        lambda batch: {
            "labels": [
                [
                    1.0 if letter in answers else 0.0
                    for letter in CHOICES
                ]
                for answers in batch["correct_answers"]
            ],
        },
        batched=True,
        batch_size=32,
    )
    return ds


def preds_to_logits(predictions: list, threshold: int = 0.5):
    sigmoid = torch.nn.Sigmoid()
    predictions = torch.Tensor(predictions)
    probs = sigmoid(predictions)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred


def multi_label_metrics(predictions, labels, threshold=0.5):
    def exact_match_score(refs, preds):
        exact_score = [p == r for p, r in zip(preds.tolist(), refs.tolist())]
        return sum(exact_score) / len(exact_score)

    def hamming_score(refs, preds):
        scores = []
        for pred, ref in zip(preds.tolist(), refs.tolist()):
            labels_pred = [id2label[i] for i, p in enumerate(pred) if p == 1]
            labels_ref  = [id2label[i] for i, r in enumerate(ref)  if r == 1]
            corrects = sum([p == r for p, r in zip(pred, ref)])
            total_r = len(list(set(labels_pred + labels_ref)))
            scores.append(corrects / total_r)
        return sum(scores) / len(scores)

    y_pred = preds_to_logits(predictions, threshold=threshold)
    # print("\nLength:", len(predictions), file=sys.stderr)
    # print("Preds:\n", predictions, file=sys.stderr)
    # print("Any non-zero in Logits?:", np.any(y_pred), file=sys.stderr)
    # print("Logits:\n", y_pred, file=sys.stderr)

    y_true = labels
    # print("Ref:\n", y_true, file=sys.stderr)

    metrics = {
        "f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "emr": exact_match_score(y_true, y_pred),
        "hamming": hamming_score(y_true, y_pred),
    }
    return metrics


def compute_metrics(pred: EvalPrediction):
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0]
    else:
        preds = pred.predictions
    result = multi_label_metrics(predictions=preds, labels=pred.label_ids)
    return result


def instance_tokenizer(
        instance: dict,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        **kwargs
) -> dict:
    """
    Tokenise instance, adding special tokens between question and answers.
    Base on Yanis Labrak DEFT 2023, updated following
    https://stackoverflow.com/a/74708120.
    """
    CLS = tokenizer.cls_token  # "<s>"
    BOS = tokenizer.bos_token  # "<s>"
    SEP = tokenizer.sep_token  # "</s>"
    EOS = tokenizer.eos_token  # "</s>"
    answers = f" {SEP} ".join([
        f"({letter}) {answer}"
        for letter, answer in instance['answers'].items()
    ])
    # CLS at the beginning and EOS/SEP at the end are added automatically by
    # the tokenizer
    text = f"{CLS} {instance['question']} {SEP} {answers} {EOS}"
    # text = f"{instance['question']} {SEP} {answers}"
    res = tokenizer(
        text, truncation=True, max_length=max_length, padding="max_length")
    return res


def main(
    model_path: str,
    train_corpus_path: str,
    dev_corpus_path: str,
    train_run_name: str,
    new_model_path: str,
    train_output_dir: str = "train_results/",
    max_length: int = 512,
    epochs: int = 1,
    report_to: str = "none",
    test_corpus_path: str = None,
    test_result_path: str = None,
    **kwargs
):
    if report_to == "wandb":
        wandb_login()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # clean_up_tokenization_spaces=True,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # Load dataset and pre-process to tokenize items
    ds = load_dataset(train_corpus_path, dev_corpus_path)
    tokenized_ds = ds.map(
        instance_tokenizer,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            **kwargs,
        },
        batched=False,
    )

    print(">>>>> BEFORE MODEL.from_pretrained", file=sys.stderr)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        problem_type="multi_label_classification",
        num_labels=len(CHOICES),
    )

    # Train model
    training_args = TrainingArguments(
        run_name=train_run_name,
        output_dir=train_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=False,

        # W&B config
        report_to=report_to,
        logging_steps=1,  # how often to log to W&B

        # From Yanis Labrak's DEFT 2023
        learning_rate=2e-5,
        greater_is_better=True,
        metric_for_best_model="emr",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,

        # From Yanis Labrak's DEFT 2023
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Save trained model
    tokenizer.save_pretrained(new_model_path)
    trainer.model.save_pretrained(new_model_path)


    # ------------------ EVALUATION ------------------
    import warnings
    warnings.warn(
        "Inference using the test dataset will be done now with the model"
        " resulting from fine-tuning. Tests have shown that using the trained"
        " model straight after fine-tuning gives better results (possibly due"
        " to the trained tokenizer not loading correctly from the saved model"
        " when running a separate inference method)."
    )

    print("Loading dataset TEST")
    test_ds = load_dataset(test_corpus_path)["train"]
    test_tokenized_ds = test_ds.map(
        instance_tokenizer,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            **kwargs,
        },
        batched=False,
    )

    print("Inference")
    predictions, labels, _ = trainer.predict(test_tokenized_ds)

    print("Prediction output")
    print("Raw logits:", predictions)
    logits = preds_to_logits(predictions)
    print("Modified logits:", logits)

    answers = [
        [
            id2label[i]
            for i, p in enumerate(instance_logits)
            if p == 1
        ]
        for instance_logits in logits
    ]
    print("Letters:", answers)

    ids_test = [d["id"] for d in test_tokenized_ds]
    results = [
        f"{id};" + "|".join(answer)
        for id, answer in zip(ids_test, answers)
    ]
    deft.write_results(results, test_result_path)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
