from transformers import GPTNeoXForCausalLM, AutoTokenizer

pythia_model = 'EleutherAI/pythia-12b-deduped'

def main(result_path: str, corpus_path: str, model: str = 'OpenAssistant/oasst-sft-1-pythia-12b', template_id: str = '0'):
    checkpoint = model

    #tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #llm = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)
    llm = GPTNeoXForCausalLM.from_pretrained(checkpoint, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def generate(input_string):
        #print('###' + input_string + '###')
        prompt = "<|prompter|>%s<|endoftext|><|assistant|>" % input_string
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = llm.generate(inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print('@@@' + generated + '@@@')
        return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id)) #, bare=True)
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
