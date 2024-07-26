import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main(result_path: str, corpus_path: str, model: str = 'HuggingFaceH4/zephyr-7b-beta', template_id: str = '0'):
    checkpoint = model 

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config=BitsAndBytesConfig(
        #load_in_8bit=True,
        # llm_int8_threshold=6.0,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
        #llm_int8_enable_fp32_cpu_offload=True, 
    )
    device_map = {
        "": 0
    }
    llm = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map, torch_dtype=torch.float16)#, load_in_8bit=True) #quantization_config=quant_config)#, load_in_8bit=True) 

    def generate(input_string):
        messages = [
            #{
            #    "role": "system",
            #    "content": "You are a friendly chatbot who responds accuractly to the user without explaining the answer.",
            #},
            {"role": "user", "content": input_string},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        #inputs = tokenizer(input_string, return_tensors="pt")
        outputs = llm.generate(encodeds.to('cuda'), max_new_tokens=32)#, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)#, pad_token_id=tokenizer.eos_token_id)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(input_string):].split('[/INST]')[1]
        return generated
        #return generated.split('<|assistant|>')[-1].split('\n')[1]
        #return generated[len(input_string):]

    import deft
    results = deft.run_inference(generate, corpus_path, deft.template_from_id(template_id))
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
