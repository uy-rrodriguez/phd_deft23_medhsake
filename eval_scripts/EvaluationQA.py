#!/usr/bin/env python3
"""Recipe for the evaluation of the question answering system of FrenchMedMCQA.
> Run the evaluation script:
    > python EvaluationQA.py --references="./references_qa.txt" --predictions="./sample_qa.txt"
Authors
 * Yanis LABRAK 2023
"""

import argparse

from sklearn.metrics import classification_report, f1_score, accuracy_score

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-r", "--references", default="./references_qa.txt", help = "Reference file")
parser.add_argument("-p", "--predictions", default="./sample_qa.txt", help = "Predictions file")
args = vars(parser.parse_args())

class SystemColors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

f_refs = open(args["references"],"r")
pairs_refs = [l.split(";") for l in f_refs.read().split("\n") if len(l) > 0]
pairs_refs = {p[0]: p[1].split("|") for p in pairs_refs}
f_refs.close()

f_preds = open(args["predictions"],"r")
pairs_preds = [l.split(";") for l in f_preds.read().split("\n") if len(l) > 0]
pairs_preds = {p[0]: p[1].split("|") for p in pairs_preds}
f_preds.close()

# Check if identifiers list are differents lengths
if len(pairs_refs) != len(pairs_preds):
    print(f"{SystemColors.FAIL} The number of identifiers doesn't match the references ! {SystemColors.ENDC}")
    exit(0)

# Check if all required identifiers are presents
if list(set([k in pairs_preds.keys() for k in pairs_refs.keys()])) != [True]:
    print(f"{SystemColors.FAIL} A required identifiers is missing ! {SystemColors.ENDC}")
    exit(0)

refs  = [pairs_refs[k] for k in pairs_refs.keys()]
preds = [pairs_preds[k] for k in pairs_refs.keys()]

def compute_accuracy_exact_match(preds, refs):
    exact_score = []
    for p, r in zip(preds, refs):
        exact_score.append(sorted(p) == sorted(r))
    return sum(exact_score) / len(exact_score)

def compute_accuracy_hamming(preds, refs):
    corrects = [True for p in preds if p in refs]
    corrects = sum(corrects)
    total_refs = len(list(set(preds + refs)))
    return corrects / total_refs

hamming_scores = [compute_accuracy_hamming(r, p) for r, p in zip(refs, preds)]
hamming_score = sum(hamming_scores) / len(hamming_scores)

exact_match = compute_accuracy_exact_match(refs, preds)

print("#"*60)
print(f"Hamming Score: {SystemColors.OKGREEN} {hamming_score} {SystemColors.ENDC}")
print(f"Exact Match Ratio: {SystemColors.OKGREEN} {exact_match} {SystemColors.ENDC}")
print("#"*60)
