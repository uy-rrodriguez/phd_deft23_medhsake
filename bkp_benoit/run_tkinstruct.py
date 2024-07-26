#https://huggingface.co/allenai/tk-instruct-large-def-pos
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main(result_path: str, corpus_path: str, model: str = 'tk-instruct-3b-def', template_id: str = "0"):
    tokenizer = AutoTokenizer.from_pretrained('allenai/' + model)
    model = AutoModelForSeq2SeqLM.from_pretrained('allenai/' + model).to('cuda')

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs, max_length=32)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id))
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
