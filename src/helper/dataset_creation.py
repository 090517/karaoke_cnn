# Imports
import numpy as np
import os.path as path
from os import makedirs, listdir
from time import strftime
from helper.audio_processing import get_snippet, mix_audio, magphase

# Pre_train_set - Only vocal tracks
# Goal: Build an internal representation and be able to reconstruct the input data
# If this is possible: start adding noise/music
def build_pre_training_data(vocal_path, out_path, file_amount=1e4, chunk_size = 1000,
                        intervals = 6, sample_rate = 48000, n_fft = 800, index_offset = 0,
                        snippets_per_file = 5, hop_factor=4, puffer = 2, *args, **kwargs):
    # Amount: Amount of file combinations got generate
    # Snippets_per_file: snippets that should be created per file
    # Chunk_size: amount of file combinations chunk
    
    # Note: Interval is now amount of sample windows and NOT duration (in s)
    # Init out path
    if not path.exists(out_path):
        makedirs(out_path)
    
    # get a list of all vocal tracks
    vocal_files = [f for f in listdir(vocal_path) if path.isfile(path.join(vocal_path, f))]
    np.random.shuffle(vocal_files)
    
    # Init FFT params
    sample_points = int(intervals * n_fft)
    print(sample_points," sample points is a",1000 * sample_points/sample_rate,"ms time window")
    hop_length = int(n_fft/hop_factor)
    t = (sample_points//hop_length) + 1
    if intervals == 1:
        truth_length = t
    else:
        truth_length = (n_fft//hop_length) + 1 + 2*puffer
        
    fft_data_type = np.dtype([('joined', np.float32, (int(n_fft/2) + 1, t)),
                              ('truth',  np.float32, (int(n_fft/2) + 1, truth_length))]) 
    print("Input data shape:", (int(n_fft/2) + 1, t))
    print("Truth shape:", (int(n_fft/2) + 1, truth_length))
    # Init Training Set 
    chunk_amount = min(chunk_size,file_amount)
    train_set = np.empty(shape=(chunk_amount * (snippets_per_file),),
                         dtype = fft_data_type)
    
    chunk_index = 1
    for i in range(int(file_amount)): 
        # randomly pick a vocal track and a music track
        vocal_file = np.random.choice(vocal_files) 
        if i%chunk_size == 0 and i!=0:
            # Store chunk
            np.random.shuffle(train_set)
            np.save(path.join(out_path,"training_set"+str(chunk_index*chunk_size+index_offset)+".npy"),
                    train_set)
            chunk_amount = min(chunk_size,file_amount-(chunk_size*chunk_index))
            if chunk_amount == 0:
                break
            chunk_index += 1
            # Init next chunk
            train_set = np.empty(shape=(chunk_amount * snippets_per_file,),
                                 dtype = fft_data_type)
    
        # Load track matrix
        vocal_track = np.load(path.join(vocal_path,vocal_file))
        
        # Pad tracks to simulate empty start/end of audio stream
        pad_amount = hop_length * (hop_factor-1)  # depends on hop length -> this is adjusted for nnft/4
        vocal_track = np.pad(vocal_track,(pad_amount,pad_amount), "constant", constant_values=0)
        
        # determine insert index and insert data
        insert_index = (i % chunk_size) * snippets_per_file
        
        # determine position of window that should be predicted
        center_index = t//2
        start_index = center_index - (hop_factor//2) - puffer # These depend on hoplength -> /4 in this case
        end_index = center_index + (hop_factor//2) + puffer
        
        for j in range(snippets_per_file):
            # get random snippets from tracks
            vocal_snippet, _ = get_snippet(vocal_track, sample_points)
        
            # join tracks and add noise
            joined_audio, vocals_only = mix_audio(vocal_snippet, np.zeros_like(vocal_snippet),
                                                  noise_ratio = 0.002)

            # Magnitude Spectrum Transform
            joined_spec, _ = magphase(joined_audio, n_fft, hop_length)
            truth_spec, _ = magphase(vocals_only, n_fft, hop_length)
        
            train_set[insert_index + j]["joined"] = joined_spec
            train_set[insert_index + j]["truth"] = truth_spec[:,start_index:end_index+1]
                     
        if i%500 == 0:
            print(i," samples created -",strftime("%H:%M:%S"))
    
    # save the final junk of joined track and vocal data
    np.save(path.join(out_path,"training_set"+str(chunk_index*chunk_size)+".npy"),
            train_set)
    
    return train_set


def build_training_data(vocal_path, music_path, out_path, file_amount=1e4, chunk_size = 1000,
                        intervals = 6, sample_rate = 48000, n_fft = 800, index_offset = 0,
                        snippets_per_file = 5, music_only = 2, hop_factor=4, puffer = 1, *args, **kwargs):
    # Amount: Amount of file combinations got generate
    # Snippets_per_file: snippets that should be created per file
    # Chunk_size: amount of file combinations chunk
    
    # Note: Interval is now amount of sample windows and NOT duration (in s)
    # Init out path
    if not path.exists(out_path):
        makedirs(out_path)
    
    # get a list of all vocal tracks
    vocal_files = [f for f in listdir(vocal_path) if path.isfile(path.join(vocal_path, f))]
    # get a list of all music tracks
    music_files = [f for f in listdir(music_path) if path.isfile(path.join(music_path, f))]
    print("Detected: ",len(vocal_files),"vocal files and ",len(music_files),"music files")
    # shuffle lists
    np.random.shuffle(vocal_files)
    np.random.shuffle(music_files)
    
    # Init FFT params
    sample_points = int(intervals * n_fft)
    print(sample_points," sample points is a",1000 * sample_points/sample_rate,"ms time window")
    hop_length = int(n_fft/hop_factor)
    t = (sample_points//hop_length) + 1
    if intervals == 1:
        truth_length = t
    else:
        truth_length = (n_fft//hop_length) + 1 + 2*puffer
    fft_data_type = np.dtype([('joined', np.float32, (int(n_fft/2) + 1, t)),
                              ('truth',  np.float32, (int(n_fft/2) + 1, truth_length))]) 
    print("Input data shape:", (int(n_fft/2) + 1, t))
    print("Truth shape:", (int(n_fft/2) + 1, truth_length))
    # Init Training Set 
    chunk_amount = min(chunk_size,file_amount)
    train_set = np.empty(shape=(chunk_amount * (snippets_per_file + music_only),),
                         dtype = fft_data_type)
    
    chunk_index = 1
    for i in range(int(file_amount)): 
        # randomly pick a vocal track and a music track
        vocal_file = np.random.choice(vocal_files) 
        music_file = np.random.choice(music_files) 
        if i%chunk_size == 0 and i!=0:
            # Store chunk
            np.random.shuffle(train_set)
            np.save(path.join(out_path,"training_set"+str(chunk_index*chunk_size+index_offset)+".npy"),
                    train_set)
            chunk_amount = min(chunk_size,file_amount-(chunk_size*chunk_index))
            if chunk_amount == 0:
                break
            chunk_index += 1
            # Init next chunk
            train_set = np.empty(shape=(chunk_amount * (snippets_per_file + music_only),),
                                 dtype = fft_data_type)
    
        # Load track matrix
        vocal_track = np.load(path.join(vocal_path,vocal_file))
        music_track = np.load(path.join(music_path,music_file))
        
        # Pad tracks to simulate empty start/end of audio stream
        pad_amount = hop_length * (hop_factor-1)  # depends on hop length -> this is adjusted for nnft/4
        vocal_track = np.pad(vocal_track,(pad_amount,pad_amount), "constant", constant_values=0)
        music_track = np.pad(music_track,(pad_amount,pad_amount), "constant", constant_values=0)
        
        # determine insert index and insert data
        insert_index = (i % chunk_size) * (snippets_per_file + music_only)
        
        # determine position of window that should be predicted
        center_index = t//2
        start_index = center_index - (hop_factor//2) - puffer # These depend on hoplength -> /4 in this case
        end_index = center_index + (hop_factor//2) + puffer
        
        for j in range(snippets_per_file):
            # get random snippets from tracks
            vocal_snippet, _ = get_snippet(vocal_track, sample_points)
            music_snippet, _ = get_snippet(music_track, sample_points)
        
            # join tracks and add noise
            joined_audio, vocals_only = mix_audio(vocal_snippet, music_snippet, *args, **kwargs)

            # Magnitude Spectrum Transform
            joined_spec, _ = magphase(joined_audio, n_fft, hop_length)
            truth_spec, _ = magphase(vocals_only, n_fft, hop_length)
        
            train_set[insert_index + j]["joined"] = joined_spec
            train_set[insert_index + j]["truth"] = truth_spec[:,start_index:end_index+1]
            
        for k in range(music_only):
            only_music_snippet, _ = get_snippet(music_track, sample_points)
            fake_vocals = np.zeros_like(only_music_snippet)
            
            joined_spec, _ = magphase(only_music_snippet, n_fft, hop_length)
            truth_spec, _ = magphase(fake_vocals, n_fft, hop_length)
            
            train_set[insert_index + snippets_per_file + k]["joined"] = joined_spec
            train_set[insert_index + snippets_per_file + k]["truth"] = truth_spec[:,start_index:end_index+1]
        
        if i%500 == 0:
            print(i," samples created -",strftime("%H:%M:%S"))
    
    # save the final junk of joined track and vocal data
    np.save(path.join(out_path,"training_set"+str(chunk_index*chunk_size)+".npy"),
            train_set)
    return train_set
