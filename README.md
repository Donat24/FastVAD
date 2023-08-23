# FastVAD
A fast VAD based on a GRU


## Why FastVAD



## How to use?

```
fast_vad_model = torch.hub.load(
    repo_or_dir  = 'Donat24/FastVAD',
    model        = 'fast_vad',
    force_reload = True
)

audio_data = torch.ones(512) # put in your 16khz audio data here 

speech_fast_vad = fast_vad_model.predict(audio_data).item()

```

## Sample Apps 


