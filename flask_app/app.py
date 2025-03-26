import importlib
import logging
import os
import sys

from datetime import datetime
from logging.config import dictConfig

from flask import Flask, request

# Trick to import local packages when this script is run from the terminal
sys.path.append(os.path.abspath("."))

from util import llm_scores


DEFAULT_MODEL = "models/llama3/llama-3-8b-deft_002_20240731"


dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        },
        'file': {
            'format': '%(message)s',
        },
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': f"logs/flask_{datetime.now().strftime('%Y%m%d-%H%M')}.log",
            'formatter': 'file',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    },
    'loggers': {
        'llm': {
            'level': 'DEBUG',
            'handlers': ['file']
        },
    }
})


app = Flask(__name__)
app._model, app._tokenizer = None, None


def reload(module_name):
    app.logger.debug(f"Reload: '{module_name}'")
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    sys.modules[module_name] = module
    globals().update(
        {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__')
        else
        {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')
    })
    # Take this opportunity to set the Flask logger in the module
    module._flask_logger = logging.getLogger("llm")
    return module


@app.route("/test")
def test():
    if not app._model or not app._tokenizer:
        return "Load the model by calling /load first", 400
    if not app._test_module or not app._test_func:
        return "Set module to be tested by calling /change_test first", 400

    module = reload(app._test_module)
    func = getattr(module, app._test_func)
    res = func(model=app._model, tokenizer=app._tokenizer)
    return res


@app.route("/change_test")
def change_test(module: str = None):
    module = module or request.args.get("module")
    if not module or len(module.split("/")) < 2:
        return "Provide 'module' to be tested as 'path.to.module/function'", 400

    module, func = module.split("/")
    app._test_module, app._test_func = module, func
    return f"Changed module under test to '{module}∕{func}'"


@app.route("/load")
def load(model: str = None):
    model = model or request.args.get("model") or DEFAULT_MODEL
    app.logger.debug(f"Loading '{model}'")
    app._model, app._tokenizer = llm_scores.load_model(model)
    return f"Model '{model}' loaded"


if __name__ == '__main__':
    # Run the app with python app.py for this to take effect!

    # Load model during startup
    load(DEFAULT_MODEL)

    # Default module to be tested
    change_test("util.llm_scores/calc_hf_perplexity")

    # Kill zombie process after an error
    # import subprocess
    # subprocess.call(["ps"])
    # subprocess.call(["kill", "-KILL", "37842"])

    app.run(host="0.0.0.0")
    # app.run(host="0.0.0.0", use_reloader=True)  # Not working in the server...
    # app.run(host="0.0.0.0", debug=True) # Too dangerous!
