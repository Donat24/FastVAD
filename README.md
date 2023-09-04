# FastVAD
A fast voice activity detection based on simple neural network with just 50k parameters.

## How to use?

```python
import torch

fast_vad_model = torch.hub.load(
    repo_or_dir  = 'Donat24/FastVAD',
    model        = 'fast_vad',
    force_reload = True
)

audio_data = torch.ones(512) # put in your 16khz audio data here 

speech_probability = fast_vad_model.predict(audio_data).item()

```

## Sample Apps 

Our GitHub repository includes [sample code](https://github.com/Donat24/FastVAD/blob/main/examples/showcase.py) that detects voice activity from the microphone and shows it on a graph.

## Performance

| Silero (ONNX) | Silero (JIT) | FastVAD DNN (JIT) | FastVAD GRU (JIT) |
|---|---|---|---|
| 0.36 ms | 2.22 ms | 0.49 ms | 0.63 ms |


## Todo:
* sample plots for the readme

