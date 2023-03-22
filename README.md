System for generating DEFT 2023 outputs from LLMs
=================================================

This repository uses git LFS for large files.
Use 'git lfs install' before cloning to retrive the binary files.

Install:
```
pip install -r requirements.txt
```

Installing bitsandbytes for llama models is a bit more [involved](https://gitlab.lis-lab.fr/cluster/wiki/-/wikis/Compiling%20bitsandbytes%20for%20int8%20inference).

See RESULTS for the exact match results on the dev.
See runs.sh for how to generate runs.

Note that external APIs require API keys. Please rename api_keys.template.py to api_keys.py and set keys you need inside.


