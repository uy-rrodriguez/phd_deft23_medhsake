"""
Script to check if several runs of a model return different logits for the same
sequence.
"""

import itertools
import json
import os
import sys

import torch
import pandas as pd
import matplotlib.pyplot as plt

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from deft import get_prompt, template_from_id, linearize_instance
from util.llm_scores import load_model


BASE_PROMPT_ONLY = False
FORCE_RELOAD = False


model_path: str = "models/llama3/llama-3-8b-deft_002_20240731"
corpus_path: str = "data/test-medshake-score.json"
output_path: str = "recipes/sequence_logits_hist.out"
figure_path: str = "recipes/sequence_logits_hist.png"

if BASE_PROMPT_ONLY:
    output_path = output_path.replace(".out", "_base.out")
    figure_path = figure_path.replace(".png", "_base.png")


choices = "a b c d e".split()
combinations = []
for i in range(1, len(choices) + 1):
    combinations.extend(itertools.combinations(choices, i))
combinations.sort(key=len)


# Avoid re-calculating de logits if an output exists
if not FORCE_RELOAD and os.path.exists(output_path):
    with open(output_path, encoding="utf8") as fp:
        all_results = json.load(fp)

# Calculate logits from scratch
else:
    print(f"Using model '{model_path}'")
    model, tokenizer = load_model(model_path)
    eos = tokenizer.eos_token


    print(f"Loading corpus '{corpus_path}")
    with open(corpus_path) as fp:
        corpus = json.load(fp)

    # Filter corpus for testing purposes
    id = "c2dc3f530e04c26e52fb50f570f73bf6cdc804ff7b65a12d215049c82f13a90a"
    corpus = filter(lambda s: s["id"] == id, corpus)
    sample = list(corpus)[0]

    # Base prompt (without answers)
    base_prompt: str = get_prompt(
        template_from_id("0"),
        sample,
        add_left_parenthesis=False,
    )
    base_prompt = base_prompt[:-1]  # Remove last space (tokenized in the answer)
    base_inputs = tokenizer(base_prompt, return_tensors="pt")


    # Loop over combinations and get logits for each new sequence
    all_results = {}
    for comb in combinations:

        # Get logits of base prompt only
        if BASE_PROMPT_ONLY:
            input_ids = base_inputs.input_ids

        # Get logits of prompt including a specific answer combination
        else:
            _, answer = linearize_instance(
                sample, included_answers=comb,
                include_full_answers=True,
                add_left_parenthesis=False,
            )

            # Add space removed from base prompt and EOS
            answer = " " + answer + eos
            answer_inputs = tokenizer(answer, return_tensors="pt")

            input_ids = torch.concat(
                (base_inputs.input_ids[0], answer_inputs.input_ids[0, 1:]),
            ).unsqueeze(0)

        # "Manually" get logits from model
        input_ids = input_ids.to("cuda")
        with torch.no_grad():
            outputs = model(input_ids)

        # Calculate log P
        log_probs = torch.tensor([
            torch.log(
                # softmax of logits at pos i-1
                #
                # Note: I assume that at each position i, logits[i] contains the
                # scores for each token in the vocabulary given the token i.
                # So, the score for the token x_i knowing x_j<i is found at
                # logits[i-1].
                torch.softmax(outputs.logits[0, i-1], dim=-1)
                # token id
                [input_ids[0, i]]
            )
            if i > 0 else 0
            for i in range(input_ids.size(1))
        ])

        # Sequence probability
        comb = " ".join(comb)
        len_prompt = base_inputs.input_ids.size(1)
        print(f"{comb}  ==>  Prompt log P: {log_probs[:len_prompt].sum()}")

        # Tokens might be repeated, so we add a counter to identify them
        tokens = [tokenizer.decode(input_ids[0, i]) for i in range(len_prompt)]
        for i in range(len_prompt):
            t = tokens[i]
            count_t = tokens.count(t)
            if count_t > 1:
                for j in range(1, count_t + 1):
                    tokens[tokens.index(t)] = f"{t} ({j})"

        all_results[comb] = {}
        for i, t in enumerate(tokens):
            logp = log_probs[i].item()
            all_results[comb][t] = logp
            # print(f"'{token}'  =  {logp}")


    # Save results to file
    with open(output_path, "w", encoding="utf8") as fp:
        json.dump(all_results, fp, indent=2, ensure_ascii=False)


# Generate histogram
df_logp = pd.DataFrame(all_results).T
print(df_logp)

fig, ax = plt.subplots(figsize=(30, 4))
fig.suptitle("Logits of base prompt")
ax.set_xlabel("Tokens")
ax.set_ylabel("Logits")
# ax.axhline(y=0, color="r", linestyle="-")
df_logp.boxplot(ax=ax)
ax.xaxis.set_tick_params(
    rotation=80, gridOn=True, grid_color="#EEEEEE", grid_dashes=(1, 2),
    grid_linewidth=1.5)
fig.savefig(figure_path, bbox_inches="tight")
