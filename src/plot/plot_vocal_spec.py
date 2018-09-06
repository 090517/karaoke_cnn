import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Import workaround for parent folder

import matplotlib.pyplot as plt  
import numpy as np
import tensorflow as tf
import librosa
from os import path

import global_settings as gs
from helper.audio_processing import magphase

np.random.seed(2)

vocal_path = "../input_24k/vocals"
interval_length = 20

datapoints = interval_length * gs.sample_rate

# Choose random music file
relevant_files = [path.join(vocal_path, x) for x in os.listdir(vocal_path) if (
    os.path.isfile(path.join(vocal_path,x)) and x.endswith(".npy"))]
selected_file = np.random.choice(relevant_files)

# Load it
audio = np.load(selected_file)

# Choose and select interval
start_index = np.random.choice(len(audio))
end_index = start_index + datapoints
audio = audio[start_index:end_index]

# Write audio snippet, just for verification
librosa.output.write_wav("../diagrams/rand_vocal_spec.wav",
                         audio,
                         sr=gs.sample_rate)

# STFT, Get Magnitude
mag, _ = magphase(audio,
                  n_fft=gs.n_fft,
                  hop_length=gs.hop_length)

# Log Magnitude
mag[mag<=0] = 1e-30
log_mag = np.log10(mag)

# Plot
tick_scale = gs.freq_res

# -- Plot Original vs Voice Prediction -- 
fig, ax1 = plt.subplots(1, 1)
aspect = 0.25

org_plot = ax1.imshow(log_mag[:1000, :], interpolation=None, cmap=plt.cm.magma, vmin=-2)
ax1.set_aspect(aspect)
ax1.invert_yaxis()

fig.colorbar(org_plot, ax=ax1, label="Amplitude (log10)")

labels = [str(int(item * tick_scale)) for item in ax1.get_yticks()]
ax1.set_yticklabels(labels)
ax1.set_ylabel("Hz")
ax1.set_xlabel("Hop (t)")

fig.set_size_inches(10,5)
fig.savefig("../diagrams/rand_vocal_spec.png", format = "png")
plt.show()
