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
    model_path = os.path.join(dirname, "pretrained", "dnn_dnn_gru_spcen_512_256.jit")
    return DNNDNNGRUModelWrapper(model_path)