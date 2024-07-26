#https://huggingface.co/facebook/opt-iml-30b
from transformers import AutoTokenizer, OPTForCausalLM
import torch

def main(result_path: str, corpus_path: str, model: str = 'galactica-30b', load_in_8bit=False, template_num: int = 0):
    checkpoint = 'facebook/' + model
    llm = OPTForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def generate(input_string):
        with torch.no_grad():
            inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
            outputs = llm.generate(inputs, max_new_tokens=32)

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.lm_templates[template_num], add_left_parenthesis=False)
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
