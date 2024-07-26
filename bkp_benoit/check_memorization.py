# implementation of MELD from https://arxiv.org/pdf/2303.13375.pdf
from api_keys import OPENAI_TOKEN

import json
import openai
import backoff
import Levenshtein

openai.api_key = OPENAI_TOKEN

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def openai_complete(prompt, model='text-davinci-003'):
    result = openai.Completion.create(model=model, prompt=prompt, temperature=0, max_tokens=64)
    return result['choices'][0]['text']

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def openai_chat(prompt, model='gpt-3.5-turbo'):
    result = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=64)
    return result['choices'][0]['message']['content']

def MELD_distance(reference, generated):
    # memorization effects levenshtein detector (MELD)
    # algorithm 1 in https://arxiv.org/pdf/2303.13375.pdf
    ops = Levenshtein.opcodes(reference, generated)
    matches = 0
    for op in ops:
        if op[0] == 'equal': 
            matches += op[2] - op[1]
    return 1 - int(round(matches / len(reference) * 100)) / 100
    # alternative implementation:
    #return 1.0 - Levenshtein.ratio(reference, generated)

def check_memorization(generator, corpus_path, output_path):
    with open(corpus_path) as fp:
        dev_corpus = json.loads(fp.read())

    sum_memorized = 0
    num_memorized = 0
    num_total = 0

    results = []
    for instance in dev_corpus:
        instance['saved_answers'] = instance['answers']
        prompt = instance['question'] + '\n' + '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items() if k in 'abc')
        reference = '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items() if k in 'de')
        generated = generator('Complete from (a) to (e).\n\n' + prompt)
        print('PROMPT:', prompt)
        print('REF:', repr(reference))
        print('HYP:', repr(generated))
        distance = MELD_distance(reference, generated)
        print('DISTANCE', distance)
        print()
        if distance <= 0.05:
            num_memorized += 1
        sum_memorized += distance
        num_total += 1 
        instance['meld'] = {'prompt': prompt, 'reference': reference, 'generated': generated, 'distance': distance}
        results.append(instance)

    print('MELD:', num_memorized / num_total)
    print('AVG DISTANCE:', sum_memorized / num_total)

    with open(output_path, 'w') as fp:
        fp.write(json.dumps(results))

def main(corpus_path: str, output_path: str, model: str = 'openai/gpt-3.5-turbo-0301', recompute_from_output=False):
    if recompute_from_output:
        with open(output_path) as fp:
            results = json.loads(fp.read())
        sum_memorized = 0
        num_memorized = 0
        num_total = 0
        for instance in results:
            prompt = instance['meld']['prompt']
            reference = instance['meld']['reference']
            generated= instance['meld']['generated']
            print('PROMPT:', prompt)
            print('REF:', repr(reference))
            print('HYP:', repr(generated))
            distance = MELD_distance(reference, generated)
            print('DISTANCE:', distance)
            print()
            if distance <= 0.05:
                num_memorized += 1
            sum_memorized += distance
            num_total += 1 
        print('MELD:', num_memorized / num_total)
        print('AVG DISTANCE:', sum_memorized / num_total)
    else:
        api, llm = model.split('/', 1)

        def generate(input_string):
            return openai_chat(input_string, llm)

        check_memorization(generate, corpus_path, output_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
