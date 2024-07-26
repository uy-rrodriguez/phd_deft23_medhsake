import re
import json
import random

lm_templates = [
'''Ceci est une question de QCM de l\'examen de pharmacie. Réponds avec la ou les lettres correspondant à la bonne réponse.\n\n%s\nRéponse(s) : (''',
#'''Corrigé du QCM de pharma.\n%s\nRéponse(s) : (''',
#'''Alice est une excellente pharmacienne. Elle répond aux questions de Pierre qui est interne en pharmacie.\nPierre : ma question est la suivante : %s\n Alice : je connais la bonne réponse et c'est (''',
#'''Correction du QCM de l\'examen de pharmacie. %s\nRéponse(s) : (''',
#'''Alice est une intelligence artificielle experte en pharmacie. Elle répond aux questions de Bob avec précision.\nBob: %s\n Alice: (''',
]
lm_shots_intro_context = '''Pour contexte, une question de QCM est une question suivie de plusieurs options identifiées avec des lettres (a), (b), (c), (d) et (e). Il est attendu que la réponse contienne uniquement la ou les lettres correspondant à la bonne réponse. '''
lm_shots_intro = [
    lm_shots_intro_context + '''Tu seras présenté avec une question de QCM et tu dois répondre uniquement avec la ou les lettres correspondant à la bonne réponse. Voici quelques examples de questions et le format de réponse attendu.\n\n%s\n\nMaintenant, tu dois suivre la consigne suivante :''',
    lm_shots_intro_context + '''Voici quelques examples de questions et le format de réponse attendu.\n\n%s''',
]
lm_shot_template = '''EXEMPLE :\n%s'''
lm_templates_en = [
'''This is a multiple choice question from the pharma exam. Reply with the letter or the letters corresponding to the correct answer.\n\n%s\n\nAnswer : (''',
]

letters = 'abcdefghijklmnopqrstuvwxyz'

def linearize_instance(instance, include_correct_answers=False, include_full_answers=False, add_left_parenthesis=False, bare=False, **kwargs):
    result = instance['question'] + '\n' + '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items())
    if bare:
        return result
    elif include_correct_answers:
        if include_full_answers:
            result += '\nRéponse(s) : ' + '; '.join('(%s) %s' % (a, instance['answers'][a]) for a in instance['correct_answers']) + '.\n'
        else:
            result += '\nRéponse(s) : ' + ' '.join('(%s)' % a for a in instance['correct_answers'])
    else:
        result += '\nRéponse(s) :' + (' (' if add_left_parenthesis else '')
    return result

#def linearize_instance(instance, include_correct_answers=False):
#    result = instance['question'] + '\n' + '\n'.join('(%s) %s.' % (k, v) for k, v in instance['answers'].items())
#    if include_correct_answers:
#        result += '\nRéponse(s) : ' + ' '.join('(%s)' % a for a in instance['correct_answers'])
#    return result

def get_random_shots(num_shots, corpus, fixed_shots_idx=None):
    """
    Helper to select random QA from the training corpus, to use as few shots.
    """
    fixed_shots_idx = fixed_shots_idx or []
    shots_idx = [i for i in fixed_shots_idx]
    shots_idx += random.choices(
        [
            i for i in range(len(corpus))
            if i not in fixed_shots_idx
        ],
        k = num_shots - len(fixed_shots_idx),
    )
    shots = []
    for i, sample in enumerate(corpus):
        if i in shots_idx:
            shots.append(sample)
        if len(shots) == num_shots:
            break
    return shots


def get_prompt(prompt, instance,
               num_shots=0, few_shots_corpus=None, fixed_shots_idx=None,
               **kwargs):
    assert num_shots == 0 or few_shots_corpus is not None, \
        "A corpus is required to randomly select the few shots"
    if num_shots > 0:
        few_shots = get_random_shots(num_shots, few_shots_corpus, fixed_shots_idx)
        shots = [
            linearize_instance(shot, bare=False, include_correct_answers=True,
                            **kwargs)
            for shot in few_shots
        ]
        return "%s\n\n%s" % (
            lm_shots_intro[1] % "\n\n".join([lm_shot_template % s for s in shots]),
            prompt % linearize_instance(instance, bare=True, **kwargs),
        )
    else:
        return prompt % linearize_instance(instance, bare=True, **kwargs)

def extract_answer(answer, num_answers=5, stop_at_line_break=False, **kwargs):
    answer = re.sub('Ceci est une question de QCM.*', '', answer).strip().lower()
    if stop_at_line_break:
      answer = re.split(r'\n[ \t]*\n', answer)[0]
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


def run_inference(generator, corpus_path, template, num_shots=0, **kwargs):
    with open(corpus_path) as fp:
        dev_corpus = json.loads(fp.read())

    num_exact_correct = 0
    num_hamming_correct = 0
    num_hamming = 0

    results = []
    all_match = []
    all_hamming = []
    for instance in dev_corpus:
        prompt = get_prompt(
            template, instance,
            num_shots=num_shots, few_shots_corpus=dev_corpus,
            **kwargs
        )
        print('PROMPT: [%s]' % prompt)
        generated = generator(prompt)
        print('GENERATED: [%s]' % generated)
        answer = extract_answer(generated, len(instance['answers']), **kwargs)
        print(answer, instance['correct_answers'])
        is_exact_match = set(answer) == set(instance['correct_answers'])
        if is_exact_match:
            num_exact_correct += 1
        hamming_val = hamming(answer, instance['correct_answers'])
        num_hamming_correct += hamming_val
        num_hamming += len(instance['answers'])
        results.append(instance['id'] + ';' + '|'.join(list(sorted(answer))))

        all_match.append(is_exact_match)
        all_hamming.append(hamming_val)

    print('EXACT MATCH:', num_exact_correct / len(dev_corpus))
    # print('HAMMING DIST:', num_hamming_correct / num_hamming)
    print('HAMMING DIST:', num_hamming_correct / len(dev_corpus))

    from util import classify_questions as cq
    emr_by_class, hamming_by_class = cq.get_average_by_difficulty(
        corpus_path,
        all_match,
        all_hamming,
    )
    print('EXACT MATCH AVG BY CLASS:', emr_by_class)
    print('HAMMING DIST AVG BY CLASS:', hamming_by_class)

    return results

def template_from_id(desc):
    if desc.startswith('en'):
        return lm_templates_en[int(desc[3:])]
    else:
        return lm_templates[int(desc)]

def write_results(results, output_path):
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(results) + '\n')
