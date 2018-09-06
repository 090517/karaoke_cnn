import tensorflow as tf
import numpy as np
import wave
import librosa

from helper.audio_processing import magphase
from helper.audio_retrieval import audio_file_to_np
import global_settings as gs

import os.path as path
from os import makedirs, listdir

def insert_entry(input_window, new_data):
    input_window[0:-len(new_data)] = input_window[len(new_data):]
    input_window[-len(new_data):] = new_data

def prepare_test_songs(test_files):
    #TODO
    # Get all files in the predestined folder
    for test_file in test_files:
        file_name = test_file.split(".")[0]
        complete_path = path.join(gs.test_in_path, test_file)
        npy_path = path.join(gs.test_npy_path, file_name+".npy")

        # Transform to np
        audio_file_to_np(complete_path, npy_path, sampling_rate = gs.sample_rate)

def predict_song(audio, sess):
    predicted_audio = []
    karaoke_audio = []
    # emulate an audio stream
    input_window = np.zeros(gs.intervals * gs.n_fft)

    # Song Prediction
    i = 0
    while i < len(audio):
        if len(audio) - i < gs.n_fft:
            new_input = np.pad(audio[i:],(0, gs.n_fft - len(audio[i:])), "constant",
                               constant_values=0)
        else:
            new_input = audio[i:i+gs.n_fft]
        insert_entry(input_window, new_input)
        i += gs.n_fft
        
        # Audio Stream to FFT
        mag, phase = magphase(input_window,
                              n_fft=gs.n_fft,  
                              hop_length=gs.hop_length)
        
        # Predict
        feed_dict_test = {gs.x: np.array([mag])}
        pred_data = sess.run(gs.pred, feed_dict_test)
        
        #print("Min,Max input:",(np.min(pred_data),np.max(pred_data)))
        pred_data[pred_data<0] = 0
        predicted_frames = pred_data[0]
        
        center_index = mag.shape[1]//2
        start_index = center_index - (gs.hop_factor//2) - gs.puffer
        end_index = center_index + (gs.hop_factor//2) + gs.puffer
        
        original_frames = np.array(mag[:,start_index:(end_index+1)])

        # -- WIP Substraction with intensity factor --
        intensity = 2.5
        karaoke_cut = original_frames - (predicted_frames * intensity)
        karaoke_cut[karaoke_cut<0] = 0

        phase_snippet = phase[:,start_index:(end_index+1)] 
        
        # FFT Prediction to audio
        reconstr_audio = librosa.istft(pred_data[0] * phase_snippet,
                                       center = True,
                                       hop_length = gs.hop_length)
        reconstr_karaoke = librosa.istft(karaoke_cut * phase_snippet,
                                      center = True,
                                      hop_length = gs.hop_length)
        
        predicted_audio += reconstr_audio.tolist()[gs.hop_length*gs.puffer:-(gs.hop_length*gs.puffer)]
        karaoke_audio += reconstr_karaoke.tolist()[gs.hop_length*gs.puffer:-(gs.hop_length*gs.puffer)]

    return predicted_audio, karaoke_audio

def predict_all_test_songs():
    test_files = [f for f in listdir(gs.test_in_path) if path.isfile(path.join(gs.test_in_path, f))]

    # Transform test songs
    prepare_test_songs(test_files)

    with tf.Session(graph=gs.model_graph, config=gs.config) as sess:
        sess.run(gs.init_op)
        gs.saver.restore(sess,path.join(gs.model_path,gs.model_name))
        gs.drop.training=False
        for test_file in test_files:
            file_name = test_file.split(".")[0]

            # Load Test songs
            audio = np.load(path.join(gs.test_npy_path, file_name+".npy"))
            predicted_audio,karaoke_audio = predict_song(audio, sess)

            if not path.exists(gs.test_pred_path):
                makedirs(gs.test_pred_path)

            librosa.output.write_wav(path.join(gs.test_pred_path,file_name+"_pred.wav"),
                                     np.array(predicted_audio),
                                     sr = gs.sample_rate)
            librosa.output.write_wav(path.join(gs.test_pred_path,file_name+"_karaoke.wav"),
                                     np.array(karaoke_audio),
                                     sr = gs.sample_rate)

            print("Finished - ",file_name)

predict_all_test_songs()
