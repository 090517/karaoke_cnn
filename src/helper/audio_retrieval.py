# Imports
import numpy as np
import os.path as path
import pafy
import audioread
import librosa
from os import makedirs

def download_audio(uris, base_path):
    if not path.exists(base_path):
        makedirs(base_path)
    
    for i,uri in enumerate(uris):
        vid = pafy.new(uri, basic=True)
        audio_data = vid.getbestaudio()
        out_file = base_path+"song_"+uri+"."+audio_data.extension
        if path.exists(out_file):
            print("URI - "+str(i+1)+" of "+str(len(uris))+":",
                  uri,"- Already downloaded")
            continue
        audio_data.download(filepath=out_file)
        print("URI - "+str(i)+" of "+str(len(uris))+":",
                  uri,"- Finished")

# Helper to assign file information
def load_file_info(file):
    if len(file)>4:
        raise TypeError("The input file has unexpected many fields:",len(file))
        
    uri, start, stop = file[0], float(file[1]), float(file[2])
    if len(file)==4:
        longplay = file[3]
    else:
        longplay = "n"
    return uri, start, stop, longplay

def audio_file_to_np(in_file_path, out_file_path, sampling_rate = 48000, *args, **kwargs):
    # Load file; if previously processed
    if path.exists(out_file_path):
        track = np.load(out_file_path)
        return track
    
    if not path.exists(path.dirname(out_file_path)):
        makedirs(path.dirname(out_file_path))
        
    # Else load original audio and store matrix for later use
    track = librosa.load(in_file_path, sampling_rate, *args, **kwargs)
    np.save(out_file_path, track[0])
    return track[0]

# Transform raw youtube audio to processable np matrices
def raw_audio_to_np(file_data, in_path, out_path, longplay_interval=240,
                    longplay_duration=60, f_endings = ["webm","m4a"], *args, **kwargs):
    # file data should be in format: uri, start(secs) , stop(secs from end), [opt. longplay]
    for file in file_data:
        uri, start_init, stop_init, longplay = load_file_info(file)
        
        # check audio file duration and adjust stop timer
        # NOTE: HARD coded file ending - This will break if they filetypes vary
        # Detect correct file extension
        f_ending = None
        for file_ending in f_endings:
            if path.exists(path.join(in_path,"song_"+uri+"."+file_ending)):
                f_ending = file_ending
        if f_ending == None:
            raise ValueError("The following file could not be found:", path.join(in_path, "song_"+uri),
                             "with extensions:",f_endings)
            
        with audioread.audio_open(path.join(in_path,"song_"+uri+"."+f_ending)) as f:
            audio_duration = f.duration
        stop_time = audio_duration - stop_init
        
        if longplay == "y":
            i = 1
            time_index = start_init
            while time_index < stop_time:
                if not path.exists(path.join(out_path,"song_"+uri+"_"+str(i)+".npy")):
                    audio_file_to_np(path.join(in_path,"song_"+uri+"."+f_ending),
                                     path.join(out_path,"song_"+uri+"_"+str(i)+".npy"),
                                     offset = time_index,
                                     duration = longplay_duration,
                                     *args, **kwargs)
                print("File written - song_"+uri+"_"+str(i)+".npy")
                i += 1
                time_index += longplay_interval 
        else:
            if not path.exists(path.join(out_path,"song_"+uri+".npy")):
                audio_file_to_np(path.join(in_path,"song_"+uri+"."+f_ending),
                                 path.join(out_path,"song_"+uri+".npy"),
                                 offset = start_init,
                                 duration = stop_time - start_init,
                                 *args, **kwargs)
            print("File written - song_"+uri+".npy")
