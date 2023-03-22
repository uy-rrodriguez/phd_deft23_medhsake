#!/usr/bin/env python3
"""Recipe for the evaluation of the classification system of FrenchMedMCQA.
> Run the evaluation script:
    > python EvaluationClassification.py --references="./references_classification.txt" --predictions="./sample_classification.txt"
Authors
 * Yanis LABRAK 2023
"""

import argparse

from sklearn.metrics import classification_report, f1_score, accuracy_score

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-r", "--references", default="./references_classification.txt", help = "Reference file")
parser.add_argument("-p", "--predictions", default="./sample_classification.txt", help = "Predictions file")
args = vars(parser.parse_args())

class SystemColors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

f_refs = open(args["references"],"r")
pairs_refs = [l.split(";") for l in f_refs.read().split("\n") if len(l) > 0]
pairs_refs = {p[0]: p[1] for p in pairs_refs}
f_refs.close()

f_preds = open(args["predictions"],"r")
pairs_preds = [l.split(";") for l in f_preds.read().split("\n") if len(l) > 0]
pairs_preds = {p[0]: p[1] for p in pairs_preds}
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

cr = classification_report(
    refs,
    preds,
    digits=4,
    zero_division=0.0,
    target_names=["1","2","3","4","5"],
)

accuracy = accuracy_score(refs, preds)
f1_macro = f1_score(refs, preds, average='macro')

print("#"*60)
print(cr)
print("#"*60)
print(f"Accuracy: {SystemColors.OKGREEN} {accuracy * 100} {SystemColors.ENDC}")
print(f"Macro F1-Score: {SystemColors.OKGREEN} {f1_macro * 100} {SystemColors.ENDC}")
print("#"*60)
