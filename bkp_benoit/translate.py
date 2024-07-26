import sys
import json
from tqdm import tqdm

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename) as fp:
    data = json.loads(fp.read())

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#facebook/nllb-200-distilled-600M
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="fra_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to('cuda')

batch_size = 20
translated = []

def add_result(i, field, value):
    while len(translated) <= i:
        translated.append({})
    if '/' in field:
        k1, k2 = field.split('/')
        if k1 not in translated[i]:
            translated[i][k1] = {}
        translated[i][k1][k2] = value
    else:
        translated[i][field] = value

def run_translation(batch, batch_info):
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to('cuda')

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=max(len(x) for x in inputs) * 2)

    detokenized = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    for (i, field), translation in zip(batch_info, detokenized):
        add_result(i, field, translation)

batch = []
batch_info = []

for i, instance in enumerate(tqdm(data, desc='translating', total=len(data))):
    for k, v in instance.items():
        if k not in ['question', 'answer']:
            add_result(i, k, v)
        else:
            add_result(i, k, None)

    batch.append(instance['question'])
    batch_info.append((i, 'question'))
    for k, v in instance['answers'].items():
        batch.append(v)
        batch_info.append((i, 'answers/' + k))

    if len(batch) >= batch_size:
        run_translation(batch, batch_info)
        batch = []
        batch_info = []
else:
    if len(batch) > 0:
        run_translation(batch, batch_info)

with open(output_filename, 'w') as fp:
    fp.write(json.dumps(translated, indent = 4))
