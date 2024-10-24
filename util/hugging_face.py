"""
Module with useful functions to access HuggingFace datasets and models.
"""

import huggingface_hub

import api_keys


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
    huggingface_hub.login(token=api_keys.HF_TOKEN)
