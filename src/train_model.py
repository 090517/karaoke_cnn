# Imports
import numpy as np
import os.path as path
import tensorflow as tf
from time import strftime

from helper.model_helper import training_loader
from helper.dataset_creation import build_training_data
import global_settings as gs

fft_train = training_loader(gs.training_path, gs.batch_size)

# -- Run Model --
gs.optimizer.learning_rate = gs.learning_rate
gs.drop.training = True

with tf.Session(graph=gs.model_graph, config=gs.config) as sess:
    sess.run(gs.init_op)

    if gs.load and path.exists(path.join(gs.model_path, gs.model_name + ".meta")):
        gs.saver.restore(sess, path.join(gs.model_path, gs.model_name))

    for i in range(gs.epochs):
        print("Epoch",i+1,"started -",strftime("%H:%M:%S"))
        max_loss, min_loss, avg_loss = -np.infty, np.infty, 0
        for j, (x_data, y_data) in enumerate(fft_train):
            feed_dict_tr = {gs.x: x_data, gs.y_true: y_data}
            _, curr_loss = sess.run([gs.optimizer, gs.loss], feed_dict=feed_dict_tr)
            max_loss = max(max_loss, np.max(curr_loss))
            min_loss = min(min_loss, np.min(curr_loss))
            avg_loss += np.average(curr_loss)
            if (j+1)%100 == 0:
                print("Max Loss:",max_loss)
                print("Min Loss:",min_loss)
                print("Avg Loss:",avg_loss/100)
                print("Chunk",j+1,"finished -",strftime("%H:%M:%S"))
                max_loss, min_loss, avg_loss = -np.infty, np.infty, 0

        # Save session after each epoch
        save_path = gs.saver.save(sess, 
                               path.join(gs.model_path, gs.model_name))
        # Logging
        print("Epoch",i+1,"finished -",strftime("%H:%M:%S"))

        if (i+1)%10 == 0:
            gs.optimizer.learning_rate *= gs.learning_rate_decay
            if gs.optimizer.learning_rate < 1e-15:
                gs.optimizer.learning_rate = 1e-15
            print("Learning rate set to: ",gs.optimizer.learning_rate)
            train_set = build_training_data(gs.vocal_path,
                                            gs.music_path,
                                            gs.training_path,
                                            file_amount=gs.file_amount,
                                            intervals=gs.intervals,
                                            chunk_size = gs.chunk_size,
                                            snippets_per_file = gs.snippets_per_file,
                                            music_only = gs.music_only,
                                            n_fft = gs.n_fft,
                                            index_offset = gs.index_offset,
                                            sample_rate = gs.sample_rate,
                                            hop_factor = gs.hop_factor,
                                            puffer = gs.puffer
                                            )
