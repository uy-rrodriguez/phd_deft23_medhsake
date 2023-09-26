System for generating DEFT 2023 outputs from LLMs
=================================================

* Update 2023-09-26: add code to finetune and perform inference with LLaMa2

The DEFT'23 shared task consists in answering pharma exam MCQs. This system converts the questions and possible answers to prompts and uses LLMs to generate answers.
The approach is described in our [paper](http://talnarchives.atala.org/CORIA-TALN/CORIA-TALN-2023/479307.pdf). It ranked 1st at the shared task.
This repository contains scripts to generate prompts, run off-the-shelf models and finetune the LLaMa models. It also contains the LoRA weights for the finetuned models. 

This repository uses git LFS for large files.
Use 'git lfs install' before cloning to retrive the binary files.

Install:
```
pip install -r requirements.txt
```

Note that bitsandbytes may need to be recompiled to support your cuda version.

See RESULTS for the exact match results on the dev.
See runs for how to generate runs.

Note that external APIs require API keys. Please rename api_keys.template.py to api_keys.py and set keys you need inside.

Please cite the follwing paper:
```
@inproceedings{Favre:CORIA-TALN:2023,
    author = "Favre, Benoit",
    title = "LIS@DEFT'23 : les LLMs peuvent-ils r\'epondre \`a des QCM ? (a) oui; (b) non; (c) je ne sais pas.",
    booktitle = "Actes de CORIA-TALN 2023. Actes du D\'efi Fouille de Textes@TALN2023",
    month = "6",
    year = "2023",
    address = "Paris, France",
    publisher = "Association pour le Traitement Automatique des Langues",
    pages = "46-56",
    note = "",
    url = "http://talnarchives.atala.org/CORIA-TALN/CORIA-TALN-2023/479307.pdf"
}
```
