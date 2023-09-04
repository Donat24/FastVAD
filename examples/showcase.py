import time
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import numpy as np
from threading import Thread,Event
import time
import torch


#Eigenes Modell
frame_length = 512
hop_length   = 256

#fast_vad_model = torch.hub.load(
#    repo_or_dir  = 'Donat24/FastVAD',
#    model        = 'fast_vad',
#    force_reload = True
#)

fast_vad_model_dnn = torch.hub.load(
    repo_or_dir  = 'Donat24/FastVAD',
    model        = 'fast_vad_dnn',
    force_reload = True
)

#PyAudio
p = pyaudio.PyAudio()

recording       = Event()
sample_rate     = 16000
numb_frames     = frame_length  * 100
input_device    = -1
frames          = np.zeros(numb_frames, dtype=np.float32)

outputs_fast_vad = np.zeros(200, dtype=np.float32)
outputs_fast_vad_dnn = np.zeros(200, dtype=np.float32)

def record():
    global recording, sample_rate, numb_frames, input_device, frames, outputs_fast_vad, outputs_fast_vad_dnn
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
        #speech_fast_vad = fast_vad_model.predict(data_tensor).item() 
        speech_fast_vad_dnn = fast_vad_model_dnn.predict(data_tensor).item() 

        #Logging
        #print(f"{speech_fast_vad} - {speech_fast_vad_dnn}")
        
        #Append Preds
        #outputs_fast_vad = np.concatenate((outputs_fast_vad, [speech_fast_vad]))
        #outputs_fast_vad = outputs_fast_vad[ - 200 : ]

        outputs_fast_vad_dnn = np.concatenate((outputs_fast_vad_dnn, [speech_fast_vad_dnn]))
        outputs_fast_vad_dnn = outputs_fast_vad_dnn[ - 200 : ]

def stop():
    recording.clear()

def live_update_demo():
    
    fig = plt.figure()
    ax              = fig.add_subplot(111, label="Waveform")
    axis_prediction = fig.add_subplot(111, label="Pred")
    axis_prediction_dnn = fig.add_subplot(111, label="Pred DNN")

    #Waveform
    ax.set_xlim([0, numb_frames])
    ax.set_ylim([-1, 1])
    (wave,) = ax.plot(frames, animated=True)
    
    axis_prediction.set_xlim([0,200])
    axis_prediction.set_ylim([0, 1])
    axis_prediction.set_xticks([])
    axis_prediction.set_yticks([])
    axis_prediction_dnn.set_xlim([0,200])
    axis_prediction_dnn.set_ylim([0, 1])
    axis_prediction_dnn.set_xticks([])
    axis_prediction_dnn.set_yticks([])
    #(pred,)         = axis_prediction.plot(outputs_fast_vad,     color= "orange", animated=True)
    (pred_dnn,) = axis_prediction_dnn.plot(outputs_fast_vad_dnn, color= "green",  animated=True)

    plt.show(block=False)
    plt.pause(0.1)

    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(wave)
    #axis_prediction.draw_artist(pred)
    axis_prediction_dnn.draw_artist(pred_dnn)
    fig.canvas.blit(fig.bbox)

    while True:

        fig.canvas.restore_region(bg)

        wave.set_ydata(frames)
        ax.draw_artist(wave)
        #pred.set_ydata(outputs_fast_vad)
        #axis_prediction.draw_artist(pred)
        pred_dnn.set_ydata(outputs_fast_vad_dnn)
        axis_prediction_dnn.draw_artist(pred_dnn)


        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        plt.pause(.001)

if __name__ == '__main__':
    Thread(target=record).start()
    live_update_demo()