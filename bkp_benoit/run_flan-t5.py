#https://huggingface.co/docs/transformers/model_doc/flan-t5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main(result_path: str, corpus_path: str, model: str = 'flan-t5-xxl', template_id: str = '0'):
  llm = AutoModelForSeq2SeqLM.from_pretrained("google/" + model).to('cuda')
  tokenizer = AutoTokenizer.from_pretrained("google/" + model)

  def generate(input_string):
      inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
      outputs = llm.generate(inputs, max_length=32)

      return tokenizer.decode(outputs[0], skip_special_tokens=True)

  import deft
  results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id))
  deft.write_results(results, result_path)

if __name__ == '__main__':
  import fire
  fire.Fire(main)
