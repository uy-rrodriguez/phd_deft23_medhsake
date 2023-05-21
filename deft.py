import re
import json

lm_templates = [
'''Ceci est une question de QCM de l\'examen de pharmacie. Réponds avec la ou les lettres correspondant à la bonne réponse.\n\n%s''',
#'''Corrigé du QCM de pharma.\n%s\nRéponse(s) : (''',
#'''Alice est une excellente pharmacienne. Elle répond aux questions de Pierre qui est interne en pharmacie.\nPierre : ma question est la suivante : %s\n Alice : je connais la bonne réponse et c'est (''',
#'''Correction du QCM de l\'examen de pharmacie. %s\nRéponse(s) : (''',
#'''Alice est une intelligence artificielle experte en pharmacie. Elle répond aux questions de Bob avec précision.\nBob: %s\n Alice: (''',
]
lm_templates_en = [
'''This is a multiple choice question from the pharma exam. Reply with the letter or the letters corresponding to the correct answer.\n\n%s\n\nAnswer : (''',
]

letters = 'abcdefghijklmnopqrstuvwxyz'

def linearize_instance(instance, include_correct_answers=False, add_left_parenthesis=False, bare=False):
    result = instance['question'] + '\n' + '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items())
    if bare:
        return result
    elif include_correct_answers:
        result += '\nRéponse(s) : ' + ' '.join('(%s)' % a for a in instance['correct_answers'])
    else:
        result += '\nRéponse(s) :' + (' (' if add_left_parenthesis else '')
    return result

#def linearize_instance(instance, include_correct_answers=False):
#    result = instance['question'] + '\n' + '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items())
#    if include_correct_answers:
#        result += '\nRéponse(s) : ' + ' '.join('(%s)' % a for a in instance['correct_answers'])
#    return result

def get_prompt(prompt, instance, few_shots=[], **kwargs):
    shots = [linearize_instance(shot, include_correct_answers=True, **kwargs) for shot in few_shots]
    return prompt % ('\n\n'.join(shots + [linearize_instance(instance, **kwargs)]),)

def extract_answer(answer, num_answers=5):
    answer = re.sub('Ceci est une question de QCM.*', '', answer).strip().lower()
    selected = re.findall(r'^[a-%s]\)|\([a-%s]\)' % (letters[num_answers - 1], letters[num_answers - 1]), answer)
    if len(selected) == 0:
        selected = re.findall(r'(\b[a-%s]\b)' % letters[num_answers - 1], answer)
    else:
        selected = [x.replace(')', '').replace('(', '') for x in selected]
    result = list(sorted(set([letter.lower() for letter in selected])))
    if len(result) == 0:
        result = ['a']
    return result

#def hamming(a, b, num):
#    A = [c.upper() if c in a else c for c in letters[:num]]
#    B = [c.upper() if c in b else c for c in letters[:num]]
#    return [x == y for x, y in zip(A, B)].count(True)

def hamming(preds, refs):
    corrects = [True for p in preds if p in refs]
    corrects = sum(corrects)
    total_refs = len(list(set(preds + refs)))
    return corrects / total_refs


def run_inference(generator, corpus_path, template, **kwargs):
    with open(corpus_path) as fp:
        dev_corpus = json.loads(fp.read())

    num_exact_correct = 0
    num_hamming_correct = 0
    num_hamming = 0

    results = []
    for instance in dev_corpus:
        prompt = get_prompt(template, instance, **kwargs)
        print(prompt)
        generated = generator(prompt)
        print(generated)
        answer = extract_answer(generated, len(instance['answers']))
        print(answer, instance['correct_answers'])
        if set(answer) == set(instance['correct_answers']):
            num_exact_correct += 1
        num_hamming_correct += hamming(answer, instance['correct_answers'])
        num_hamming += len(instance['answers'])
        results.append(instance['id'] + ';' + '|'.join(list(sorted(answer))))

    print('EXACT MATCH:', num_exact_correct / len(dev_corpus))
    print('HAMMING DIST:', num_hamming_correct / num_hamming)

    return results

def template_from_id(desc):
    if desc.startswith('en'):
        return lm_templates_en[int(desc[3:])]
    else:
        return lm_templates[int(desc)]

def write_results(results, output_path):
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(results) + '\n')

