"""
Module with useful functions to access HuggingFace datasets and models.
"""

def silent(callback):
    """
    Decorator to silence standard output. Useful during login.
    """
    def wrapper():
        import os
        import sys
        out = sys.stdout
        nul = open(os.devnull, 'w')
        sys.stdout = nul
        value = callback()
        sys.stdout = out
        return value
    return wrapper


@silent
def hf_login():
    from huggingface_hub import login
    from api_keys import HF_TOKEN
    login(token=HF_TOKEN)
