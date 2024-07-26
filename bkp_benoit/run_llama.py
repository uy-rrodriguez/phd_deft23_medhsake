from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import transformers
import torch
import sys
import json
from argparse import Namespace

from deft import linearize_instance, extract_answer, hamming

# stop inference when a given token was generated (for example '\n')
class TokenStopper(transformers.StoppingCriteriaList):
    def __init__(self, token, prompt_lengths):
        self.token = tokenizer.encode(token)[-1]
        self.prompt_lengths = prompt_lengths

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for sequence, length in zip(input_ids, self.prompt_lengths):
            sequence = sequence[length:]
            if (sequence == self.token).sum() == 0:
                return False
        else:
            return True

def run_batch(batch, eval_stats, output_fp):
    prompts = [config.prompt % linearize_instance(instance, add_left_parenthesis=True) for instance in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to('cuda')
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs, max_new_tokens=128, 
            generation_config=generation_config, 
            #stopping_criteria=[TokenStopper('\n', [len(sequence) for sequence in inputs['input_ids']])]
        )
    for instance, PROMPT, length, sequence in zip(batch, prompts, lengths, generation_output):
        generated = tokenizer.decode(sequence, skip_special_tokens=True)
        print(generated)
        answer = extract_answer(generated[len(PROMPT):].split('\n')[0])
        print(answer, instance['correct_answers'])
        output_fp.write(instance['id'] + ';' + '|'.join(sorted(answer)).lower() + '\n')

        eval_stats['num_emr'] += 1
        if set(answer) == set(instance['correct_answers']):
            eval_stats['num_emr_correct'] += 1
        eval_stats['num_hamming_sum'] += hamming(answer, instance['correct_answers'])
        eval_stats['num_hamming'] += len(instance['answers'])
    
def main(model_path : str, output_path : str, corpus_path : str = '../../json/dev.json', adapted : bool = True, batch_size : int = 1):
    global config
    import json
    with open('%s/training_config.json' % model_path) as fp:
        config = Namespace(**json.loads(fp.read()))

    #config.prompt = "This is a MCQ from the biology exam in French. Answer with the correct set of letters.\n\n%s"

    global tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(config.backbone, padding_side='left')
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    global model
    model = LlamaForCausalLM.from_pretrained(
        config.backbone,
        load_in_8bit=config.is_int8,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if adapted:
        # device_map={"": 0} fix device = None error https://github.com/huggingface/peft/issues/115
        model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.float16, device_map={"": 0}) 

    with open(corpus_path) as fp:
        corpus = json.loads(fp.read())

    eval_stats = {
        'num_emr': 0,
        'num_emr_correct': 0,
        'num_hamming_sum': 0,
        'num_hamming': 0,
    }

    with open(output_path, 'w') as output_fp:
        for i in range(0, len(corpus), batch_size):
            run_batch(corpus[i: i + batch_size], eval_stats, output_fp)

    print('EXACT MATCH:', config.output_path, eval_stats['num_emr_correct'] / eval_stats['num_emr'])
    print('HAMMING DIST:', config.output_path, eval_stats['num_hamming_sum'] / eval_stats['num_hamming'])

if __name__ == '__main__':
    import fire
    fire.Fire(main)
