import time
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import numpy as np
from threading import Thread,Event
import time
import librosa
import torch


#Eigenes Modell
frame_length = 512
hop_length   = 512

fast_vad_model = torch.hub.load(
    repo_or_dir  = 'Donat24/FastVAD',
    model        = 'fast_vad',
    force_reload = True
)

#Silero
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
    model        = 'silero_vad',
    force_reload = True,
    onnx         = True,
)

#PyAudio
p = pyaudio.PyAudio()

recording       = Event()
sample_rate     = 16000
numb_frames     = frame_length  * 100
input_device    = -1
frames          = np.zeros(numb_frames, dtype=np.float32)

outputs_fast_vad = np.zeros(100, dtype=np.float32)
outputs_silero   = np.zeros(100, dtype=np.float32)

def record():
    global recording, sample_rate, numb_frames, input_device, frames, outputs_fast_vad, outputs_silero
    stream = p.open(
        format   = pyaudio.paFloat32,
        channels = 1,
        rate     = sample_rate,
        input    = True,
        frames_per_buffer = 0,
        input_device_index = input_device, 
    )

    while True:
        
        #Read Data
        data = np.frombuffer(stream.read(hop_length), dtype=np.float32)
        frames  = np.concatenate((frames, data))
        frames  = frames[ - numb_frames : ]
        data_tensor = torch.from_numpy(frames[ - frame_length : ].copy())
        
        #Predict
        speech_fast_vad = fast_vad_model.predict(data_tensor).item() 
        speech_silerio  = silero_model(data_tensor, sample_rate).item()

        #Logging
        print(f"{speech_fast_vad}\t{speech_silerio}")
        
        #Append Preds
        outputs_fast_vad = np.concatenate((outputs_fast_vad, [speech_fast_vad]))
        outputs_fast_vad = outputs_fast_vad[ - 100 : ]
        outputs_silero = np.concatenate((outputs_silero, [speech_silerio]))
        outputs_silero = outputs_silero[ - 100 : ]

def stop():
    recording.clear()

def live_update_demo():
    
    fig = plt.figure()
    ax              = fig.add_subplot(111, label="Waveform")
    axis_prediction = fig.add_subplot(111, label="Pred")
    axis_prediction_silero = fig.add_subplot(111, label="Pred Silero")

    #Waveform
    ax.set_xlim([0, numb_frames])
    ax.set_ylim([-1, 1])
    (wave,) = ax.plot(frames, animated=True)
    
    axis_prediction.set_xlim([0,100])
    axis_prediction.set_ylim([0, 1])
    axis_prediction.set_xticks([])
    axis_prediction.set_yticks([])
    axis_prediction_silero.set_xlim([0,100])
    axis_prediction_silero.set_ylim([0, 1])
    axis_prediction_silero.set_xticks([])
    axis_prediction_silero.set_yticks([])
    (pred,)         = axis_prediction.plot(outputs_fast_vad,     color= "orange", animated=True)
    (pred_silero,) = axis_prediction_silero.plot(outputs_silero, color= "green",  animated=True)

    plt.show(block=False)
    plt.pause(0.1)

    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(wave)
    axis_prediction.draw_artist(pred)
    axis_prediction_silero.draw_artist(pred_silero)
    fig.canvas.blit(fig.bbox)

    while True:

        fig.canvas.restore_region(bg)

        wave.set_ydata(frames)
        ax.draw_artist(wave)
        pred.set_ydata(outputs_fast_vad)
        axis_prediction.draw_artist(pred)
        pred_silero.set_ydata(outputs_silero)
        axis_prediction_silero.draw_artist(pred_silero)


        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        plt.pause(.001)

if __name__ == '__main__':
    Thread(target=record).start()
    live_update_demo()