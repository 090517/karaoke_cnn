from helper.model import build_modelgraph
from helper.model_helper import training_loader
import tensorflow as tf

model_name = "karaoke"

# Paths
training_path = "../training_24k_center/"
pre_training_path = "../pre_training_24k_center/"
model_path = "../model_data_24k_center/"
ccpath = "../ccmixter_corpus/"
music_path = "../input_24k/music/"
vocal_path = "../input_24k/vocals/"
music_uri_path = "../urls/music.csv"
vocal_uri_path = "../urls/vocals.csv"
music_yt_path = "../raw/music/"
vocals_yt_path = "../raw/vocals/"
test_in_path = "../test_music/originals"
test_npy_path = "../test_music/npy"
test_pred_path = "../test_music/predictions"

# Training Globals
file_amount = 5000
intervals = 3
chunk_size = 500
snippets_per_file = 4
music_only = 1
n_fft = 8192
hop_factor = 8
hop_length = int(n_fft/hop_factor)
index_offset = 0
sample_rate = 24000
max_freq = sample_rate//2 
puffer = hop_factor//2

total_snippets = (snippets_per_file + music_only) * file_amount
snippet_duration = (intervals * n_fft)/sample_rate
total_duration = total_snippets*snippet_duration
hours = (total_duration//60)//60
minutes = (total_duration//60)%60
secs = total_duration%60
amount_bins = n_fft//2
freq_res = max_freq/amount_bins

print("This will produce:",total_snippets,"snippets")
print("A single snippet lasts:",snippet_duration,"s")
print("Lowest detectable frequency:",1/(snippet_duration/5),"Hz")
print("Total duration",total_duration,"s")
print("1 Iteration will produce audio data with length:",int(hours),"h",int(minutes),"m", int(secs),"s")
print("Frequency Resolution:", freq_res,"Hz/bin")

# Pre-Training Globals
pre_epochs = 50
pre_batch_size = 5
pre_file_amount = 2000
pre_chunk_size = 1000
pre_snippets_per_file = 3

# Training Globals
epochs = 100
batch_size = 10
learning_rate = 1e-9
learning_rate_decay = 0.5
load = True

# Model Structure
in_shape = (4097, 25)

cnn_shape = [(64,(3,3),(3,1),"same",tf.nn.tanh),
             (32,(3,3),(1,1),"same",tf.nn.relu),
             (64,(5,1),(5,1),"same",tf.nn.relu),
             (64,(3,1),(3,1),"same",tf.nn.relu),
             (32,(3,3),(1,1),"same",tf.nn.relu),
             (64,(3,3),(3,1),"same",tf.nn.relu),
             (32,(3,3),(1,1),"same",tf.nn.relu)
            ]

dense_shape = [(2048, tf.nn.tanh),
               (2048, tf.nn.relu),
               (5863, tf.nn.relu)]  

dense_t_reshape = 13 # reshape "columns" / time slots of the dense layer

dcnn_shape = [(32,(5,5),(1,1),"valid",tf.nn.relu),
              (32,(5,5),(3,1),"same",tf.nn.relu),
              (32,(5,5),(3,1),"same",tf.nn.relu),
              (1,(3,1),(1,1),"valid",tf.nn.relu)]

out_shape = (4097, 17)

model_graph, saver, important_layers, init_op = build_modelgraph(
    in_shape, out_shape, cnn_shape, dense_shape, dcnn_shape, dense_t_reshape, learning_rate=learning_rate)
x, y_true, pred, loss, optimizer, drop = important_layers

# init gpu settings
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
