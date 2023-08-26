dependencies = ["numpy", "torch", "torchaudio"]

import torch
import os
import numpy as np
from utils.model_wrapper import *

def fast_vad():
    """
    returns a classifier
    """
    dirname    = os.path.dirname(__file__)
    model_path = os.path.join(dirname, "pretrained", "gru_spcen_512_512.jit")
    return FastVadModelWrapper(model_path)