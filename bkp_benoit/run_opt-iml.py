#https://huggingface.co/facebook/opt-iml-30b
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main(result_path: str, corpus_path: str, model: str = 'opt-iml-30b', template_num: int = 0):
    checkpoint = 'facebook/' + model
    llm = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = llm.generate(inputs, max_new_tokens=32)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.lm_templates[template_num])
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
