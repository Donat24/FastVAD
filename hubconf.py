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
    model_path = os.path.join(dirname, "pretrained", "dnn_gru_mfcc_512_512.jit")
    return ModelWrapperDNNGRU(model_path)
    #model_path = os.path.join(dirname, "pretrained", "dnn_dnn_gru_mfcc_512_512.jit")
    #return ModelWrapperDNNDNNGRU(model_path)