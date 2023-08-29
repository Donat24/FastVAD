import torch
import numpy as np

class ModelWrapper():
    """
    wrapper for the jitted Model
    """
    def __init__(self, path) -> None:
        self.model = torch.jit.load(path)
        self.model.eval()
        self.reset()

    def reset(self):
        """
        resets the context. should be used before classifying a new file
        """
        pass
    
    def buffer_to_tensor(self, buffer):
        return torch.from_numpy(np.frombuffer(buffer, dtype=np.float32))
    
    def predict_buffer(self, buffer):
        """
        input is a buffer (f.x. pyaudio stream)
        """
        return self.predict(data=self.buffer_to_tensor(buffer))

    def predict(self, data):
        """
        input is a PyTorch tensor
        """
        pass

class GRUModelWrapper(ModelWrapper):
    """
    wrapper for the main model architekture
    """
    def __init__(self, path) -> None:
        self.model = torch.jit.load(path)
        self.reset()

    def reset(self):
        self.context_1    = None
        self.context_2    = None
    
    def buffer_to_tensor(self, buffer):
        return torch.from_numpy(np.frombuffer(buffer, dtype=np.float32))

    def predict(self, data):
        speech, self.context_1, self.context_2 = self.model(data, self.context_1, self.context_2)
        return speech

class DNNDNNModelWrapper(ModelWrapper):
    """
    wrapper for the main model architekture
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

    def predict(self, data):
        speech, self.context_1, self.context_2, self.context_3 = self.model(data, self.context_1, self.context_2, self.context_3)
        return speech

class DNNDNNGRUModelWrapper(ModelWrapper):
    """
    wrapper for the main model architekture
    """
    def __init__(self, path) -> None:
        self.model = torch.jit.load(path)
        self.reset()

    def reset(self):
        self.context_1    = None
        self.context_2    = None
        self.context_3    = None
        self.context_4    = None
    
    def buffer_to_tensor(self, buffer):
        return torch.from_numpy(np.frombuffer(buffer, dtype=np.float32))

    def predict(self, data):
        speech, self.context_1, self.context_2, self.context_3, self.context_4 = self.model(data, self.context_1, self.context_2, self.context_3, self.context_4)
        return speech