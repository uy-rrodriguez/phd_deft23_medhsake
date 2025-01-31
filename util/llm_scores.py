"""
Scripts that use advanced features of LLMs, like accessing the internal weights
and token scores/probabilities.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

# HuggingFace authentication
# from util.hugging_face import hf_login
# hf_login()

def print_err(*args, **kwargs):
    kwargs.update({"file": sys.stderr})
    print(*args, **kwargs)


def load_model(
        model_path: str,
        use_special_pad_token: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if use_special_pad_token:
        tokenizer.add_special_tokens(
            {"pad_token": "<|reserved_special_token_250|>"})
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def generate(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        input_string: str,
        max_new_tokens: int = 32,
        return_raw_output: bool = False,
        # Optional parameters to the model's "generate" method
        print_hyper_kwargs: bool = False,
        **hyper_kwargs
):
    # Hyper-parameters
    hyper_kwargs = {
        k: v
        for k, v in hyper_kwargs.items()
        if v is not None
    }
    if print_hyper_kwargs and len(hyper_kwargs):
        print_err(
            "Hyper-parameters:",
            *[f"  - {k}: {v}" for k, v in hyper_kwargs.items()],
            "\n",
            sep="\n",
        )

    inputs = tokenizer(input_string, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        # pad_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.eos_token_id,

        # Handle optional parameters
        # https://huggingface.co/docs/transformers/en/generation_strategies
        **hyper_kwargs
    )

    if not return_raw_output:
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]
    return inputs, outputs


def output_scores(model, tokenizer):
    """
    https://www.perplexity.ai/search/how-to-get-output-score-from-t-55XxhfhpSaOh_MnUxFHXtA
    """

    # 1 Correct Answer
    # 5987fa6bffd499eb439c90679d7fbca822d62bc639d1b9c94c68ae20e46f6004
    # ---
    # input_text = (
    #     "Ceci est une question de QCM de l'examen de pharmacie. Réponds avec la"
    #     " ou les lettres correspondant à la bonne réponse."
    #     "\n\nParmi les propositions suivantes, indiquer celle qui est exacte."
    #     " Dans les conditions physiologiques, le pH le plus élevé est mesuré"
    #     " dans:"
    #     "\n(a) Le suc gastrique."
    #     "\n(b) La bile vésiculaire."
    #     "\n(c) Le suc pancréatique."
    #     "\n(d) La salive."
    #     "\n(e) Les sécrétions intestinales."
    #     "\nRéponse(s) : ("
    # )

    # 3 Correct Answers
    # 4c0a40502de05e79aacd7131e714319e80300f37a119a944516fbde8e1d006c4
    # ---
    answers = {
        "a": "L'alcool déshydrogénase",
        "b": "L'aldéhyde déshydrogénase",
        "c": "La catalase",
        "d": "Les cytochromes P450 (voie MEOS)",
        "e": "Les flavines mono-oxygénases",
    }
    answers_comb = "a b c".split()

    def answer_to_str(letter):
        return f'({letter}) {answers[letter]}'

    newl = "\n"
    input_text = (
        "Ceci est une question de QCM de l'examen de pharmacie. Réponds avec la"
        " ou les lettres correspondant à la bonne réponse."
        "\n\nParmi les propositions suivantes, lesquelles sont exactes? Le"
        " métabolisme de l'éthanol en acétaldéhyde est catalysé par :"
        "\n"
        f"{newl.join(answer_to_str(c) for c in answers)}"
        "\nRéponse(s) : "
        f"{'; '.join(answer_to_str(c) for c in answers_comb)}"
        ".\n"
    )

    letters = list(answers.keys())
    letters_tokens = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in letters
    ]

    inputs, outputs = generate(
        model, tokenizer,
        input_text,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        # temperature=0,
        # Custom parameter
        return_raw_output=True,
    )
    # inputs        => transformers.tokenization_utils_base.BatchEncoding
    # outputs       => transformers.generation.utils.GenerateDecoderOnlyOutput
    #   attributes  => sequences, scores, logits, attentions, hidden_states

    # Squeeze output (removing first dimension of size 1)
    sequences = outputs.sequences.squeeze(0)
    # outputs.sequences => Tensor, shape: [1, 122] == [batch_size, total tokens]
    #   -> squeezed     => [122] == [total tokens = prompt + generated]

    # Get the logits
    logits = [t.squeeze(0) for t in outputs.logits]
    # logits        => tuple, len: nbr generated tokens <= max_new_tokens
    # logits[0]     => Tensor, shape: [1, 128256] == [batch_size, vocab_size]
    #   -> squeezed => [128256] == [vocab_size]

    # Get the scores
    scores = [t.squeeze(0) for t in outputs.scores]
    # scores        => tuple, len: nbr generated tokens
    # scores[0]     => Tensor, shape: [1, 128256], squeezed: [128256]

    # Convert scores to probabilities
    probs = [torch.softmax(t, dim=-1) for t in scores]
    logits_probs = [torch.softmax(t, dim=-1) for t in logits]

    # Get the generated text
    prompt_text = tokenizer.decode(sequences[:inputs.input_ids.shape[1]])
    generated_tokens = sequences[inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens)

    print_err()
    print_err(f"PROMPT: [{prompt_text}]")
    print_err(f"GENERATED: [{generated_text}]")
    print_err(f"Number of score tensors: {len(scores)}")

    print_err("\n")
    chosen_tokens_probs = [
        f"{tokenizer.decode(token_id)}  >>>  {p[token_id].item()}"
        for p, token_id in zip(
            probs,
            generated_tokens
        )
    ]
    print_err("Token probabilities")
    json.dump(chosen_tokens_probs, indent=2, fp=sys.stderr)

    print_err("\n")
    print_err("Probability of EOS")
    # n_gen = len(scores)
    # eos_results = [
    #     [probs[i][tokenizer.eos_token_id].item() for i in range(n_gen)],
    #     [logits_probs[i][tokenizer.eos_token_id].item() for i in range(n_gen)]
    # ]
    eos_results = [
        probs[0][tokenizer.eos_token_id].item(),
        logits_probs[0][tokenizer.eos_token_id].item(),
    ]
    print_err(json.dumps(eos_results, indent=2))

    # Highest tokens
    print_err("\n")
    print_err("Scores and logits of N highest tokens")
    n_high = 5
    sorted_logits, indices = logits_probs[0].sort(descending=True)
    high_results = {
        tokenizer.decode(t): (probs[0][t].item(), logit.item())
        for t, logit in zip(indices[:n_high], sorted_logits[:n_high])
    }
    print_err(json.dumps(high_results, indent=2))

    # Scores and logits of answers each time an answer token appears in the
    # response
    print_err("\n")
    print_err("Scores and logits of all answers given")
    all_answer_results = {}
    for i, token in enumerate(generated_tokens):
        if token in letters_tokens:
            all_answer_results[i] = {
                c: (probs[i][t].item(), logits_probs[i][t].item())
                for c, t in zip(letters, letters_tokens)
            }
    print_err(json.dumps(all_answer_results, indent=2))

    # return all_answer_results
    return eos_results


def main_output_scores():
    model_path = "models/llama3/llama-3-8b-deft_002_20240731"
    model, tokenizer = load_model(model_path)
    output_scores(model, tokenizer)


if __name__ == "__main__":
    import fire
    fire.Fire(main_output_scores)
