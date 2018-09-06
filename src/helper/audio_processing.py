# Imports
import numpy as np
import librosa

# Add noise to "pure" signal
def add_noise(input_signal,noise_ratio = 0.1):
    noise = np.random.rand(input_signal.shape[0]) - 0.5
    noise = noise.astype(input_signal.dtype)
    out = input_signal * (1-noise_ratio) + noise * noise_ratio
    return out

# Get random snippet
def get_snippet(track, sample_points, restricted_area=None):
    # track: numpy array, containing normalized (in [-1,1])  data of audio track
    # sample_points: amount of sample points
    # restricted_area: tuple, start and end sample point where snippet should be placed
    
    if restricted_area == None: 
        start = 0
        end = len(track)
    else:
        start = restricted_area[0]
        end = restricted_area[1]
    snippet_start = np.random.randint(start,end-sample_points)
    snippet_end = snippet_start + sample_points
    return track[snippet_start:snippet_end], (snippet_start,snippet_end)

def magphase(audio, n_fft, hop_length):
    # Transform output wave with sfft and get magnitude spectrum + phase
    fft = librosa.stft(audio,
                       center = True,
                       n_fft = n_fft,
                       hop_length = hop_length)
    mag_spec, phase = librosa.magphase(fft)
    return mag_spec, phase

# Join two tracks and add noise
def mix_audio(track_1, track_2, ratio = 0.60, deviation = 0.15,  noise_ratio = 0.005):
    # track1, track2: numpy arrays, containing normalized (in [-1,1]) data of audio tracks of same length
    # ratio: float, defining the mean influence of track 1 when mixing the tracks
    #                0 - only track 2; 1 - only track 1
    # deviation: float, controls the area where the ratio of track 1 and track 2 can fluctuate 
    # noise ratio: float, defining the max influence of noise when mixing joined_tracks and noise
    #                0 - only mixed tracks; 1 - only noise
    #              the noise value will be chosen randomly between [0,noise_ratio]
    
    assert(track_1.shape == track_2.shape)
    true_track_ratio = np.random.normal(loc = ratio, scale=deviation)
    true_track_ratio = np.clip(true_track_ratio,0,1)
    true_noise_ratio = np.random.rand() * noise_ratio
    
    mixed_tracks = track_1 * true_track_ratio + track_2 * (1-true_track_ratio)
    if true_noise_ratio != 0:
        mixed_tracks = add_noise(mixed_tracks, noise_ratio = true_noise_ratio)
    return mixed_tracks, (track_1 * true_track_ratio)
