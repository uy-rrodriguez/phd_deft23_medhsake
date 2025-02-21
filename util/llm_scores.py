"""
Scripts that use advanced features of LLMs, like accessing the internal weights
and token scores/probabilities.
"""

import itertools
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from deft import get_prompt, template_from_id, linearize_instance

# HuggingFace authentication
# from util.hugging_face import hf_login
# hf_login()


# The _flask_logger is set by the Flask app if the code is run from there.
# Otherwise the logs are printed to standard output.
_flask_logger = None
def log(*args, **kwargs):
    if _flask_logger is not None:
        _flask_logger.info(*args)
    else:
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
        log(
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

    log(f"\nPROMPT: [{prompt_text}]")
    log(f"GENERATED: [{generated_text}]")
    log(f"Number of score tensors: {len(scores)}")

    chosen_tokens_probs = [
        f"{tokenizer.decode(token_id)}  >>>  {p[token_id].item()}"
        for p, token_id in zip(
            probs,
            generated_tokens
        )
    ]
    log("\nToken probabilities")
    json.dump(chosen_tokens_probs, indent=2, fp=sys.stderr)

    log("\nProbability of EOS")
    # n_gen = len(scores)
    # eos_results = [
    #     [probs[i][tokenizer.eos_token_id].item() for i in range(n_gen)],
    #     [logits_probs[i][tokenizer.eos_token_id].item() for i in range(n_gen)]
    # ]
    eos_results = [
        probs[0][tokenizer.eos_token_id].item(),
        logits_probs[0][tokenizer.eos_token_id].item(),
    ]
    log(json.dumps(eos_results, indent=2))

    # Highest tokens
    log("\n")
    log("Scores and logits of N highest tokens")
    n_high = 5
    sorted_logits, indices = logits_probs[0].sort(descending=True)
    high_results = {
        tokenizer.decode(t): (probs[0][t].item(), logit.item())
        for t, logit in zip(indices[:n_high], sorted_logits[:n_high])
    }
    log(json.dumps(high_results, indent=2))

    # Scores and logits of answers each time an answer token appears in the
    # response
    log("\nScores and logits of all answers given")
    all_answer_results = {}
    for i, token in enumerate(generated_tokens):
        if token in letters_tokens:
            all_answer_results[i] = {
                c: (probs[i][t].item(), logits_probs[i][t].item())
                for c, t in zip(letters, letters_tokens)
            }
    log(json.dumps(all_answer_results, indent=2))

    # return all_answer_results
    return eos_results


def calc_sample_scores(
        model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
        inst: dict, combinations: list[tuple], prompt_tpl: str = "0",
):
    """
    Queries the model with the given sample `inst` to get the probabilties for
    each combinations of answers.
    """
    def answer_to_str(letter):
        return f'({letter}) {inst["answers"][letter]}'

    log(f"{'-'*80}\n{inst['id']}")

    base_prompt: str = get_prompt(
        template_from_id(prompt_tpl),
        inst,
        add_left_parenthesis=False,
    )

    results = {}
    eos_logits = []
    for comb in combinations:
        log(f"\nCOMBINATION: {' '.join(comb)}")

        prompt = (
            f"{base_prompt}"
            f"{'; '.join(answer_to_str(c) for c in comb)}"
            ".\n"
        )
        inputs, outputs = generate(
            model, tokenizer,
            prompt,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            # temperature=0.01,  # default: 0.6. score = logit / temp

            # Custom parameter
            return_raw_output=True,
        )

        # Squeeze output (removing first dimension of size 1)
        sequences = outputs.sequences.squeeze(0)
        # Get the logits
        logits = [t.squeeze(0) for t in outputs.logits]
        eos_logit = logits[0][tokenizer.eos_token_id].item()
        # Get the scores
        scores = [t.squeeze(0) for t in outputs.scores]
        eos_score = scores[0][tokenizer.eos_token_id].item()

        # Get the generated text
        prompt_text = tokenizer.decode(sequences[:inputs.input_ids.shape[1]])
        log(f"PROMPT: [{prompt_text}]")
        generated_text = tokenizer.decode(sequences[inputs.input_ids.shape[1]:])
        log(f"GENERATED: [{generated_text}]")

        # Convert scores to probabilities
        logits_probs = [torch.softmax(t, dim=-1) for t in logits]
        eos_prob = logits_probs[0][tokenizer.eos_token_id].item()
        scores_probs = [torch.softmax(t, dim=-1) for t in scores]
        eos_score_prob = scores_probs[0][tokenizer.eos_token_id].item()
        log(
            f"EOS: logit={eos_logit} | score={eos_score}"
            f" | logit_prob={eos_prob}% | score_prob={eos_score_prob}%"
        )
        raise ValueError

    results = {
        " ".join(comb): logit
        for comb, logit in zip(combinations, eos_logits)
    }
    log(f"\nALL EOS: {results}")
    return results


def calc_model_scores(
        model_path: str = "models/llama3/llama-3-8b-deft_002_20240731",
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        corpus_path: str = "data/test-medshake-score.json",
        output_path: str =
            "output/model_scores/llama3/"
            "llama-3-8b-deft_002_20240731-logits.json",
):
    """
    Queries the model with all possible combinations of answers and calculates
    their probabilities with the internal model scores.
    """
    if model is not None and tokenizer is not None:
        log(f"Using model '{model.__class__.__name__}'")
    else:
        log(f"Using model '{model_path}'")
        model, tokenizer = load_model(model_path)

    print(f"Loading corpus '{corpus_path}")
    with open(corpus_path) as fp:
        corpus = json.load(fp)

    # For debugging
    # id = "4c0a40502de05e79aacd7131e714319e80300f37a119a944516fbde8e1d006c4"
    # corpus = filter(lambda s: s["id"] == id, corpus)

    choices = "a b c d e".split()
    combs = []
    for i in range(1, len(choices) + 1):
        combs.extend(itertools.combinations(choices, i))
    combs.sort(key=len)

    all_results = {}
    for inst in corpus:
        eos_results = calc_sample_scores(model, tokenizer, inst, combs)
        all_results[inst["id"]] = eos_results

    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    return all_results


def calc_sample_logp(
        model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
        inst: dict, combinations: list[tuple], prompt_tpl: str = "0",
) -> float:
    """
    Calculates the log probability for a given sentence.

    Given S a sequence of k possible answers w_1 w_2 ... w_k, we want to
    calculate P(w_1...w_k EOS|prompt) where w_1 ... w_k are the possible answers
    (a, b, ..., ab, ac, ..., abcde).

    Log Probability is then defined as:

        log P(w_1...w_k EOS|prompt) =
            log(P(EOS|prompt, w_1 ... w_k))
            + P(w_k|prompt, w_1 ... w_{k-1})
            + ... + log(P(w_1|prompt))

    This can be interpreted as the probability given by the model to the
    sequence of tokens of a given answer.

    https://blog.uptrain.ai/decoding-perplexity-and-its-significance-in-llms/
    """
    log(f"{'-'*80}\n{inst['id']}")

    base_prompt: str = get_prompt(
        template_from_id(prompt_tpl),
        inst,
        add_left_parenthesis=False,
    )
    # Remove last space from base prompt (after "Réponse(s) :")
    # This space will be added as part of the answer, otherwise the first letter
    # is tokenized as "(a" while the rest are tokenized as single characters,
    # e.g. "b".
    base_prompt = base_prompt[:-1]

    # Get logits from the model
    base_inputs = tokenizer(base_prompt, return_tensors="pt")
    len_prompt = base_inputs.input_ids.size(1) - 1  # Ignore BOS

    results = {}
    for comb in combinations:
        log(f"COMBINATION: {' '.join(comb)}")

        letters = list(inst["answers"].keys())
        eos = tokenizer.eos_token

        # Get logits from the model
        _, answer = linearize_instance(
            inst, included_answers=comb,
            include_full_answers=True,
            add_left_parenthesis=False,
        )
        # Add space removed from base prompt and EOS
        answer = " " + answer + eos
        answer_inputs = tokenizer(answer, return_tensors="pt")

        # Join tokens of base prompt and answers
        input_ids = torch.concat(
            (base_inputs.input_ids[0], answer_inputs.input_ids[0, 1:]),
        ).unsqueeze(0).to("cuda")

        # "Manually" get logits from model based on sequence
        with torch.no_grad():
            outputs = model(input_ids)
        # outputs.logits.shape => (batch, seq length, vocab size)

        # Our algorithm differs from model.compute_transition_scores:
        # - compute_transition_scores returns the score of each token in a
        #   sequence at each generation step.
        # - compute_transition_scores expects output scores as returned by
        #   model.generate, i.e.: Tuple of len generated sequence, where each
        #   Tensor is of shape (batch size, vocab size).
        # - It's not useful for us because we need the logits of all the tokens
        #   at each generation step, so we can calculate softmax over them.
        #
        # transition_scores = model.compute_transition_scores(
        #     input_ids, [t for t in outputs.logits.reshape(input_ids.size(1), 1, -1)],
        # )
        # log(transition_scores)
        # log(transition_scores.shape)

        # Remove batch dimension
        input_ids = input_ids.squeeze(0)
        logits: torch.Tensor = outputs.logits.squeeze(0)

        log_probs = torch.tensor([
            torch.log(
                # softmax of logits at pos i-1
                #
                # Note: I assume that at each position i, logits[i] contains the
                # scores for each token in the vocabulary given the token i.
                # So, the score for the token x_i knowing x_j<i is found at
                # logits[i-1].
                torch.softmax(logits[i-1], dim=-1)
                # token id
                [input_ids[i]]
            )
            for i in range(1, input_ids.size(0))  # ignore first token BOS
        ])

        log_probs_letters = {}
        for i in range(1, input_ids.size(0)):
            prob = log_probs[i-1].item()
            token_id = input_ids[i]
            token = tokenizer.decode(token_id)

            # Search probability of answer letters
            found = False
            if i >= len_prompt and token_id != tokenizer.eos_token_id:
                token_ = token.strip()
                for t in letters:
                    _prev = tokenizer.decode(input_ids[i-1])
                    _next = tokenizer.decode(input_ids[i+1])
                    prev_is_par = _prev.endswith("(")
                    next_is_par = _next.startswith(")")
                    found = (
                        token_ == t and prev_is_par and next_is_par
                        or token_ == f"({t}" and next_is_par
                        or token_ == f"{t})" and prev_is_par
                        or token_ == f"({t})"
                    )
                    if found:
                        # log(tokenizer.decode(token_id) + "  <==")
                        log_probs_letters[t] = prob
                        letters.remove(t)
                        break

            if found:
                log(f"{token} ({token_id}) => {prob}  <<<")
            else:
                log(f"{token} ({token_id}) => {prob}")

        # Make sure we found all letters in the current combination
        assert list(comb) == sorted(log_probs_letters), (
            "Not all letters in the current combination were captured."
            f" Expected: {comb}. Found {log_probs_letters}."
        )

        # Sequence probability
        prompt_logp = log_probs[:len_prompt].sum().item()
        seq_logp = log_probs[len_prompt:].sum().item()
        eos_logp = log_probs[-1].item()
        letters_logp = sum(log_probs_letters.values()) + eos_logp
        log(f"Sequence log P: {seq_logp}")
        log(f"Prompt log P: {prompt_logp}")
        log(f"EOS log P: {eos_logp}")
        log(f"Letters + EOS log P: {letters_logp}")
        log("\n" + "*"*80 + "\n")

        results[" ".join(comb)] = {
            "seq_logp": seq_logp,
            "prompt_logp": prompt_logp,
            "eos_logp": eos_logp,
            "letters_logp": letters_logp,
            "letters": log_probs_letters,
            "seq_len": input_ids.size(0) - len_prompt,
            "prompt_len": len_prompt,
        }

    return results


def calc_model_logp(
        model_path: str = "models/llama3/llama-3-8b-deft_002_20240731",
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        corpus_path: str = "data/test-medshake-score.json",
        output_path: str =
            "output/model_scores/llama3/llama-3-8b-deft_002_20240731-logp_20250218.json",
):
    """
    Queries the model with all possible combinations of answers and calculates
    the Negative Log Likelihood of each sentence.
    """
    if model is not None and tokenizer is not None:
        log(f"Using model '{model.name_or_path}'")
    else:
        log(f"Using model '{model_path}'")
        model, tokenizer = load_model(model_path)

    log(f"Loading corpus '{corpus_path}")
    with open(corpus_path) as fp:
        corpus = json.load(fp)

    # For debugging
    # id = "4c0a40502de05e79aacd7131e714319e80300f37a119a944516fbde8e1d006c4"
    # id = "c2dc3f530e04c26e52fb50f570f73bf6cdc804ff7b65a12d215049c82f13a90a"
    # corpus = filter(lambda s: s["id"] == id, corpus)

    choices = "a b c d e".split()
    combs = []
    for i in range(1, len(choices) + 1):
        combs.extend(itertools.combinations(choices, i))
    combs.sort(key=len)

    all_results = {}
    for inst in corpus:
        sample_results = calc_sample_logp(model, tokenizer, inst, combs)
        all_results[inst["id"]] = sample_results

    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    return all_results


def hf_perplexity(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sequence: str,
) -> torch.Tensor:
    """
    Perplexity calculated following HuggingFace's algorithm, based on the
    entropy loss of a sequence.

    https://huggingface.co/docs/transformers/perplexity
    """
    max_length = model.config.max_length
    stride = max_length

    encodings = tokenizer(sequence, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    # Process the entire sequence at once (gives lower perplexity)
    max_length = seq_len

    log(f"max_length={max_length}, stride={stride}, seq_len={seq_len}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        log(f"begin_loc={begin_loc}, end_loc={end_loc}, trg_len={trg_len}")
        # log(f"{input_ids}, {target_ids}")

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over
            # valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels,
            # because it internally shifts the labels to the left by 1.
            neg_log_likelihood = outputs.loss

            # NLL can be 0 if the length of input_ids is 1
            if neg_log_likelihood.isnan():
                neg_log_likelihood = torch.tensor(0)

            log(f"neg_log_likelihood = {neg_log_likelihood}")

        # Accumulate the total negative log-likelihood and the total number of
        # tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        # log(f"{num_valid_tokens} - {batch_size} = {num_loss_tokens}")

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    log(f"exp({avg_nll}) = {ppl}")
    return ppl


def calc_hf_perplexity(
        model_path: str = "models/llama3/llama-3-8b-deft_002_20240731",
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        corpus_path: str = "data/test-medshake-score.json",
        output_path: str =
            "output/model_scores/llama3/llama-3-8b-deft_002_20240731-perp_20250219.json",
):
    """
    Uses HuggingFace evaluate metric "perplexity" for each possible combination
    of answers, for each sample in the corpus.
    """
    if model is not None and tokenizer is not None:
        log(f"Using model '{model.name_or_path}'")
    else:
        log(f"Using model '{model_path}'")
        model, tokenizer = load_model(model_path)

    print(f"Loading corpus '{corpus_path}")
    with open(corpus_path) as fp:
        corpus = json.load(fp)

    # For debugging
    # id = "4c0a40502de05e79aacd7131e714319e80300f37a119a944516fbde8e1d006c4"
    # id = "c2dc3f530e04c26e52fb50f570f73bf6cdc804ff7b65a12d215049c82f13a90a"
    # corpus = filter(lambda s: s["id"] == id, corpus)

    choices = "a b c d e".split()
    combs = []
    for i in range(1, len(choices) + 1):
        combs.extend(itertools.combinations(choices, i))
    combs.sort(key=len)

    all_results = {}
    for inst in corpus:
        log("*" * 80)
        log(inst["id"])
        results = {}
        base_prompt: str = get_prompt(
            template_from_id("0"),
            inst,
            add_left_parenthesis=False,
        )
        for comb in combs:
            log("-" * 80)
            log(comb)
            _, answer = linearize_instance(
                inst, included_answers=comb,
                include_full_answers=True,
                add_left_parenthesis=False,
            )
            prompt = base_prompt + answer + tokenizer.eos_token
            ppl = hf_perplexity(model, tokenizer, prompt)
            results[" ".join(comb)] = ppl.item() if not ppl.isnan() else "-Inf"
        all_results[inst["id"]] = results

    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    return all_results


def main(method_name: str, *args, **kwargs):
    from util import llm_scores
    method = getattr(llm_scores, method_name)
    if not method:
        raise f"Method '{method_name}' not found"
    return method(*args, **kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
