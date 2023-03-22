#https://huggingface.co/bigscience/bloomz-7b1-mt
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(result_path: str, corpus_path: str, model: str = 'bloomz-7b1-mt', template_id: str = '0'):
    checkpoint = 'bigscience/' + model

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    llm = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = llm.generate(inputs, max_new_tokens=32)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id))
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
