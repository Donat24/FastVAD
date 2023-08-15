import torch
import numpy as np

class ModelWrapper():
    """
    wrapper for the jitted Model
    """
    def __init__(self, path) -> None:
        self.model = torch.jit.load(path)
        self.reset()

    def reset(self):
        self.context_1    = None
        self.context_2    = None
        self.context_3    = None
    
    def buffer_to_tensor(self, buffer):
        return torch.from_numpy(np.frombuffer(buffer, dtype=np.float32))
    
    def predict_buffer(self, buffer):
        return self.predict(data=self.buffer_to_tensor(buffer))

    def predict(self, data):
        speech, self.context_1, self.context_2, self.context_3 = self.model(data, self.context_1, self.context_2, self.context_3)
        return speech