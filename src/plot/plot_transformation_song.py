import sys
import os
# Import workaround for parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt  
import numpy as np
import tensorflow as tf
import librosa

from helper.plotting import *
from helper.audio_processing import *
from global_settings import *

class sim_audio_stream(object):
    def insert_entry(self, new_data):
        new_data = np.array(new_data)
        if new_data.shape[0] != self.n_fft:
            raise ValueError("Invalid input dimensions:",new_data.shape,"needs to have", self.n_fft,",")
        self.input_window[0:-len(new_data)] = self.input_window[len(new_data):]
        self.input_window[-len(new_data):] = new_data
    def __init__(self, audio_data, n_fft, intervals):
        if intervals % 2 == 0:
            raise ValueError("Specify an odd interval for such that there is a center element.")
        self.audio = audio_data
        self.n_fft = n_fft
        self.intervals = intervals
        self.input_window = np.zeros(intervals * n_fft)
    def __iter__(self):
        i = 0
        self.input_window = np.zeros(intervals * n_fft)
        while i < len(self.audio):
            # Simulate the new incoming audio data and insert it
            if len(self.audio) - i < self.n_fft:
                # When remaining audio stream shorter than the interval -> fill with zeros
                new_input = np.pad(self.audio[i:],(0, self.n_fft - len(self.audio[i:])), "constant",
                                   constant_values=0)
            else:
                # Else take the next part of the audio
                new_input = self.audio[i:i+self.n_fft]
            self.insert_entry(new_input)
            i += self.n_fft
            yield np.copy(self.input_window)
        # After inserting all audio frames, push them to the center element
        for _ in range(self.intervals//2):
            self.insert_entry(np.zeros(self.n_fft))
            yield np.copy(self.input_window)

plt.style.use('seaborn-deep')

song_interval = np.array([100, 110]) * sample_rate

# chunk it into 0.5s bits and pad with zeroes if necessary
# audio = np.load("J:/karaoke_data/src/test_music/npy/Jenifer_Avila_-_01_-_El_Tranva.npy")
# audio = np.load("J:/karaoke_data/src/test_music/npy/copperhead_-_I_Got_a_Girl.npy")
audio = np.load("J:/karaoke_data/src/test_music/npy/Paper_Navy_-_08_-_Swan_Song.npy")
#audio = np.load("J:/karaoke_data/src/test_music/npy/US_Girls_-_The_Island_Song.npy")

audio = audio[song_interval[0]:song_interval[1]]

stream = sim_audio_stream(audio, n_fft = n_fft, intervals = intervals)

predicted_audio = []
karaoke_audio = []
# emulate an audio stream
input_window = np.zeros(intervals * n_fft)

with tf.Session(graph=model_graph, config=config) as sess:
    sess.run(init_op)
    saver.restore(sess, "../model_data_24k_center/karaoke")
    drop.training=False

    for input_window in stream:

        # Audio Stream to FFT
        mag, phase = magphase(input_window,
                              n_fft=n_fft,  
                              hop_length=hop_length)
        #print("Min,Max input:",(np.min(mag),np.max(mag)))
        
        # Predict
        feed_dict_test = {x: np.array([mag])}
        pred_data = sess.run(pred,feed_dict_test)
        prediction = pred_data[0]
        
        # Prevent negative predictions
        #print("Min,Max input:",(np.min(pred_data),np.max(pred_data)))
        prediction[prediction<0] = 0
        
        center_index = mag.shape[1]//2
        start_index = center_index - (hop_factor//2) - puffer # These depend on hoplength -> /4 in this case
        end_index = center_index + (hop_factor//2) + puffer
        true_mag = mag[:,start_index : (end_index+1)]

        assert(true_mag.shape == prediction.shape)
        amp = 1
        karaoke_cut = true_mag - (prediction * amp)
        karaoke_cut[karaoke_cut<0] = 0

        phase_snippet = phase[:,start_index:(end_index+1)] 
        
        # FFT Prediction to audio
        reconstr_audio = librosa.istft(prediction * phase_snippet,
                                       center = True,
                                       hop_length = hop_length)
        reconstr_karaoke = librosa.istft(karaoke_cut * phase_snippet,
                                      center = True,
                                      hop_length = hop_length)
        
        predicted_audio += reconstr_audio.tolist()[hop_length*puffer:-(hop_length*puffer)]
        karaoke_audio += reconstr_karaoke.tolist()[hop_length*puffer:-(hop_length*puffer)]

stream_buffer = intervals//2 * n_fft
streamed_audio = []
for window in stream:
    streamed_audio += window[stream_buffer:-stream_buffer].tolist()
streamed_audio = np.array(streamed_audio)
assert(len(streamed_audio) == len(predicted_audio))

print("Shape Input Audio:",len(streamed_audio))
print("Shape Output Audio:",len(predicted_audio))
print("Shape Output Audio:",len(karaoke_audio))

librosa.output.write_wav("../test_snippet_pred.wav",
                         np.array(predicted_audio),
                         sr=sample_rate)
librosa.output.write_wav("../test_snippet_karaoke.wav",
                         np.array(karaoke_audio),
                         sr=sample_rate)

orig_mag, _ = magphase(streamed_audio,
                      n_fft=n_fft,  
                      hop_length=hop_length)

pred_mag, _ = magphase(np.array(predicted_audio),
                      n_fft=n_fft,  
                      hop_length=hop_length)

karaoke_mag, _ = magphase(np.array(karaoke_audio),
                      n_fft=n_fft,  
                      hop_length=hop_length)

print("Prediction Min Max", np.amin(pred_mag),"-",np.amax(pred_mag))
print("Truth Min Max", np.amin(orig_mag),"-",np.amax(orig_mag))
print("Karaoke Min Max", np.amin(karaoke_mag),"-",np.amax(karaoke_mag))

'''
# Plot mag distribution
flat_prediction = joined_pred_spec.flatten()
print("Min. pred value:",np.amin(flat_prediction))
print("Max. pred value:",np.amax(flat_prediction))
plt.hist(flat_prediction, bins = "auto")
# x1,x2,y1,y2 = plt.axis()
plt.axis((0,500,0,120))
plt.show()
'''

tick_scale = freq_res

# -- Plot Original vs Voice Prediction -- 
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
aspect = 0.5

# Fix to plot properly
orig_mag[orig_mag<=0] = 1e-30
pred_mag[pred_mag<=0] = 1e-30
karaoke_mag[karaoke_mag<=0] = 1e-30

log_prog = np.log10(orig_mag)
log_prediction = np.log10(pred_mag)
log_karaoke = np.log10(karaoke_mag)

max_value = (max(np.amax(log_prog), np.amax(log_prediction)))
print("Max Orig.",np.amax(log_prog),"Max Pred.",np.amax(log_prediction),"Max Karaoke",np.amax(log_karaoke))

print("Truth Spec Shape:",log_prog.shape)
print("Pred Spec Shape:",log_prediction.shape)
print("Karaoke Spec Shape:",log_karaoke.shape)

org_plot = ax1.imshow(log_prog[:1000,:], vmin = -2, vmax=max_value, interpolation = None, cmap=plt.cm.magma)
ax1.set_aspect(aspect)
ax1.set_title("Input Data")
ax1.invert_yaxis()

pred_plot = ax2.imshow(log_prediction[:1000,:], vmin = -2, vmax=max_value, interpolation = None, cmap=plt.cm.magma)
ax2.set_aspect(aspect)
ax2.set_title("Voice Prediction")
ax2.invert_yaxis()

fig.colorbar(pred_plot, ax=(ax1,ax2), pad=0.05, label="Amplitude (log10)")

labels = [str(int(item * tick_scale)) for item in ax1.get_yticks()]
ax1.set_yticklabels(labels)
ax1.set_ylabel("Hz")
ax1.set_xlabel("Hop (t)")
ax2.set_xlabel("Hop (t)")

fig.set_size_inches(10,5)
fig.savefig("../diagrams/pred_comp.png", format="png")
plt.show()

# -- Plot Original vs Karaoke -- 
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
aspect = 0.5

ax1.imshow(log_prog[:1000,:], vmin = -2, vmax=max_value, interpolation = None, cmap=plt.cm.magma)
ax1.set_aspect(aspect)
ax1.set_title("Input Data")
ax1.invert_yaxis()

ax2.imshow(log_karaoke[:1000, :], vmin=-2, vmax=max_value,
           interpolation=None, cmap=plt.cm.magma)
ax2.set_aspect(aspect)
ax2.set_title("Karaoke Prediction")
ax2.invert_yaxis()

fig.colorbar(pred_plot, ax=(ax1,ax2), pad=0.05, label="Amplitude (log10)")

labels = [str(int(item * tick_scale)) for item in ax1.get_yticks()]
ax1.set_yticklabels(labels)
ax1.set_ylabel("Hz")
ax1.set_xlabel("Hop (t)")
ax2.set_xlabel("Hop (t)")

fig.set_size_inches(10,5)
fig.savefig("../diagrams/karaoke_comp.png", format="png")
plt.show()
