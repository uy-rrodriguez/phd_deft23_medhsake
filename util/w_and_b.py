"""
Module with useful functions to access Weights & Biases via HuggingFace.
"""

import os

import wandb

import api_keys
from util.hugging_face import silent


@silent
def wandb_login():
    os.environ["WANDB_PROJECT"] = "deft2023"
    # os.environ["WANDB_DEBUG"] = "true"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    wandb.login(key=api_keys.WANDB_TOKEN, verify=True)
