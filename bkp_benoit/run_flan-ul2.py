
#https://huggingface.co/google/flan-ul2
# pip install accelerate transformers bitsandbytes
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

def generate_flan_ul2(input_string):
    inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=32)

    return tokenizer.decode(outputs[0])

def main(result_path: str, corpus_path: str, template_num: int = 0):
    import deft
    results = deft.run_inference(generate_flan_ul2, corpus_path, deft.lm_templates[template_num])
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
