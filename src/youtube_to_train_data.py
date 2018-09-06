#Imports
import csv
from helper.audio_retrieval import raw_audio_to_np, download_audio
import global_settings as gs

vocal_files = []
with open(gs.vocal_uri_path,"r") as f_voc:
    vocals_reader = csv.reader(f_voc, delimiter=',', quotechar='"')
    first_line_vocal = next(vocals_reader)
    for row in vocals_reader:
        vocal_files.append(row)

music_files = []
with open(gs.music_uri_path,"r") as f_music:
    music_reader = csv.reader(f_music, delimiter=',', quotechar='"')
    first_line_music = next(music_reader)
    for row in music_reader:
        music_files.append(row)

vocal_uris = [x[0] for x in vocal_files]
music_uris = [x[0] for x in music_files]

# Download Audio
download_audio(vocal_uris, gs.vocals_yt_path)
download_audio(music_uris, gs.music_yt_path)

# Transform Audio into Training Data
raw_audio_to_np([x[:3] for x in vocal_files],
                in_path = gs.vocals_yt_path,
                out_path = gs.vocal_path,
                sampling_rate = 24000)

raw_audio_to_np([x[:4] for x in music_files],
                in_path= gs.music_yt_path,
                out_path = gs.music_path,
                sampling_rate = 24000)
