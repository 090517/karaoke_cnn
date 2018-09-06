import os
import os.path as path
import numpy as np
import audioread
import librosa

import global_settings as gs

# Get root folder content
for directory in os.listdir(gs.ccpath):
    if directory == "README":
        continue
        
    song_path = path.join(gs.ccpath,directory)

    # Open each folder and extract
    # Source-1 = Music
    # Source-2 = Vocals
    music_in_path = path.join(song_path,"source-01.wav")
    vocal_in_path = path.join(song_path,"source-02.wav")

    # open files and transform to npy representation
    conv_name = "song_"+directory+".npy"
    music_out_path = path.join(gs.music_path, conv_name)
    vocal_out_path = path.join(gs.vocal_path, conv_name)

    if not path.exists(path.join(gs.music_path,conv_name)):
        # load original audio and store matrix for later use
        track = librosa.load(music_in_path, 24000)
        np.save(music_out_path, track[0])
        print("Created:",path.join(gs.music_path,conv_name))
    else:
        print("Already created:",path.join(gs.music_path,conv_name))

    if not path.exists(path.join(gs.vocal_path,conv_name)):
        # load original audio and store matrix for later use
        track = librosa.load(vocal_in_path, 24000)
        np.save(vocal_out_path, track[0])
        print("Created:",path.join(gs.vocal_path,conv_name))
    else:
        print("Already created:",path.join(gs.vocal_path,conv_name))