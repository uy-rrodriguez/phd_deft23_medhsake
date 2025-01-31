Simple Flask app to play around with models. It loads the model and tokenizer
in memory once, and keeps a server open so the test function can be run
multiple times without the need to reload the model every time.

The test code can be changed on the fly and the app endpoint called again
without requiring to reload the entire app.

This is to avoid long waiting times due to models being load onto memory, during
quick tests.

Usage:

Run locally with:
    python app.py
    or flask --app app.py run

Run in the Slurm environment (console prompt located at the project root) with:
    bash sbatch.sh slurm_adhoc.sh flask_app/app.py

Query from the terminal with:
    wget <localhost or machine name>:5000/load -q -O -
    wget <localhost or machine name>:5000/test -q -O -