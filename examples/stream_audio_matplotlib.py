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
context_1    = None
context_2    = None
context_3    = None
frame_length = 512
hop_length   = 512
model = torch.jit.load(r"./pretrained/dnn_dnn_gru_mfcc_512_512.jit")

#Silerio
#silerio_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#    model        = 'silero_vad',
#    force_reload = True,
#    onnx         = False
#)

#PyAudio
p = pyaudio.PyAudio()

recording       = Event()
sample_rate     = 16000
numb_frames     = frame_length  * 100
input_device    = -1
frames          = np.zeros(numb_frames, dtype=np.float32)
outputs         = np.zeros(100, dtype=np.float32)
#outputs_silerio = np.zeros(200, dtype=np.float32)

def record():
    global recording, sample_rate, numb_frames, input_device, frames, context_1, context_2, context_3, outputs
    stream = p.open(
        format   = pyaudio.paFloat32,
        channels = 1,
        rate     = sample_rate,
        input    = True,
        frames_per_buffer = 0,
        input_device_index = input_device, 
    )

    while True:
        
        data = np.frombuffer(stream.read(hop_length), dtype=np.float32)
        #speech_silerio     = silerio_model(data_tensor, sample_rate).item()
        frames  = np.concatenate((frames, data))
        frames  = frames[ - numb_frames : ]
        data_tensor = torch.from_numpy(frames[ - frame_length : ].copy())
        speech, context_1, context_2 = model(data_tensor, context_1, context_2, context_3)
        print(speech.item())
        outputs = np.concatenate((outputs, [speech.item()]))
        outputs = outputs[ - 100 : ]
        
        #outputs_silerio = np.concatenate((outputs_silerio, [speech_silerio]))
        #outputs_silerio = outputs_silerio[ - 100 : ]

def stop():
    recording.clear()

def live_update_demo():
    
    fig = plt.figure()
    ax              = fig.add_subplot(111, label="Waveform")
    axis_prediction = fig.add_subplot(111, label="Pred")

    #Waveform
    ax.set_xlim([0, numb_frames])
    ax.set_ylim([-1, 1])
    (wave,) = ax.plot(frames, animated=True)
    
    axis_prediction.set_xlim([0,100])
    axis_prediction.set_ylim([0, 1])
    axis_prediction.set_xticks([])
    axis_prediction.set_yticks([])
    (pred,)         = axis_prediction.plot(outputs,         color= "orange", animated=True)
    #(pred_silerio,) = axis_prediction.plot(outputs_silerio, color= "green",  animated=True)

    plt.show(block=False)
    plt.legend()
    plt.pause(0.1)

    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(wave)
    axis_prediction.draw_artist(pred)
    #axis_prediction.draw_artist(pred_silerio)
    fig.canvas.blit(fig.bbox)

    while True:

        fig.canvas.restore_region(bg)

        wave.set_ydata(frames)
        ax.draw_artist(wave)
        pred.set_ydata(outputs)
        #pred_silerio.set_ydata(outputs_silerio)
        axis_prediction.draw_artist(pred)
        #axis_prediction.draw_artist(pred_silerio)

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        plt.pause(.001)

if __name__ == '__main__':
    Thread(target=record).start()
    live_update_demo()