import importlib
import os
import sys

from flask import Flask

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util import llm_scores


MODEL = "models/llama3/llama-3-8b-deft_002_20240731"


app = Flask(__name__)
app._model, app._tokenizer = None, None


def reload(module_name):
    # print(f"Reload: '{module_name}'", file=sys.stderr)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    sys.modules[module_name] = module
    globals().update(
        {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__')
        else
        {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')
    })


@app.route('/test')
def test():
    print("Test")
    if not app._model or not app._tokenizer:
        return "Load the model by calling /load first", 400
    reload("util.llm_scores")
    res = llm_scores.output_scores(app._model, app._tokenizer)
    # res = output_scores(app._model, app._tokenizer)
    return res


@app.route('/load')
def load():
    print(f"Loading '{MODEL}'")
    app._model, app._tokenizer = llm_scores.load_model(MODEL)
    return "Model loaded"


if __name__ == '__main__':
    load()
    app.run(host="0.0.0.0")
    # app.run(host="0.0.0.0", use_reloader=True)
    # app.run(host="0.0.0.0", debug=True)
