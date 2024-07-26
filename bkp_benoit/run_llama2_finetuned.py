#https://huggingface.co/bigscience/bloomz-7b1-mt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import finetune_llama2

def main(result_path: str, corpus_path: str, base_model: str, lora_model: str, template_id: str = '0'):
    checkpoint = 'llama2-weights/convert/' + base_model

    llm, tokenizer = finetune_llama2.load_for_inference(checkpoint, lora_model)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    def generate(input_string):
        inputs = tokenizer(input_string, return_tensors="pt")
        outputs = llm.generate(input_ids=inputs.input_ids.to('cuda'), 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.eos_token_id,
            #generation_config=generation_config,
            temperature=0,
            #do_sample=True,
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id), stop_at_line_break=True)
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
