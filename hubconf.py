dependencies = ["numpy", "torch", "torchaudio"]

import torch
import os
import numpy as np
from util.model_wrapper import ModelWrapper

def fast_vad():
    """
    returns a classifier
    """
    dirname    = os.path.dirname(__file__)
    model_path = os.path.join(dirname, "pretrained", "dnn_dnn_gru_mfcc_512_512.jit")
    return ModelWrapper(model_path)