import sys
import os
# Import workaround for parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import global_settings as gs
from helper.audio_processing import magphase
import numpy as np
import matplotlib.pyplot as plt
import librosa


audio = librosa.load("../diagrams/audio_test.wav", gs.sample_rate)[0]

# STFT, Get Magnitude
mag, _ = magphase(audio,
                  n_fft=gs.n_fft,
                  hop_length=gs.hop_length)

# Log Magnitude
mag[mag <= 0] = 1e-30
log_mag = np.log10(mag)

# Plot
tick_scale = gs.freq_res

# -- Plot Original vs Voice Prediction --
fig, ax1 = plt.subplots(1, 1)
aspect = 0.05

org_plot = ax1.imshow(log_mag[:2000, :], vmin=-2, interpolation=None, cmap=plt.cm.magma)
ax1.set_aspect(aspect)
ax1.invert_yaxis()

fig.colorbar(org_plot, ax=ax1, label="Amplitude (log10)")

labels = [str(int(item * tick_scale)) for item in ax1.get_yticks()]
ax1.set_yticklabels(labels)
ax1.set_ylabel("Hz")
ax1.set_xlabel("Hop (t)")

fig.set_size_inches(7, 5)
fig.savefig("../diagrams/consonants.png", format="png")
plt.show()
