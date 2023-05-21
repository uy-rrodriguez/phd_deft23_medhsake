#https://huggingface.co/bigscience/bloomz-7b1-mt
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  trust_remote_code=True
)

def main(result_path: str, corpus_path: str, model: str = 'mpt-7b-instruct', template_id: str = '0'):
    checkpoint = 'mosaicml/' + model

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    llm = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to('cuda')

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = llm.generate(inputs, max_new_tokens=32)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):].split('#')[0]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id))
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
