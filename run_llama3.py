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
    prompt_template_id: str = "0",
    num_shots: int = 0,
    shots_full_answer: bool = False,
):

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

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
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
