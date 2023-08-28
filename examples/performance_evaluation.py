import cProfile
import torch
import timeit
import statistics as st

iterations = 10000

fast_vad_model = torch.hub.load(
    repo_or_dir  = 'Donat24/FastVAD',
    model        = 'fast_vad',
    force_reload = True
)

silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
    model        = 'silero_vad',
    force_reload = True,
    onnx         = True,
)


def benchmark(inference_function):

    audio_data = torch.ones(512) # mockup 16khz audio data     

    result = timeit.repeat(lambda: inference_function(audio_data),repeat=20, number=iterations)

    mean = st.mean(result)
    print(f"mean time for {iterations} iteratiions is {mean} s")
    print(f"standard deviation over 10 runs is {st.pstdev(result)} s")    
    print(f"mean time for 1 inference step is {(mean / iterations) * 1000 } ms")


print("Benchmark Silero")
benchmark(lambda x: silero_model(x,16000))

print("\n")
print("Benchmark Fast VAD")
benchmark(lambda x: fast_vad_model.predict(x))




