# Imports
import numpy as np
import os.path as path
import tensorflow as tf
from time import strftime

import global_settings as gs
from helper.model_helper import training_loader

# -- Run Pre-Training --
gs.optimizer.learning_rate = gs.learning_rate
gs.drop.training = True

fft_pre_train = training_loader(gs.pre_training_path, gs.pre_batch_size)

with tf.Session(graph=gs.model_graph, config=gs.config) as sess:
    sess.run(gs.init_op)

    if gs.load and path.exists(path.join(gs.model_path,gs.model_name + ".meta")):
        gs.saver.restore(sess, path.join(gs.model_path, gs.model_name))

    for i in range(gs.pre_epochs):
        print("Epoch",i+1,"started -",strftime("%H:%M:%S"))
        max_loss, min_loss, avg_loss = -np.infty, np.infty, 0
        for j, (x_data, y_data) in enumerate(fft_pre_train):
            feed_dict_tr = {gs.x: x_data, gs.y_true: y_data}
            _, curr_loss = sess.run(
                [gs.optimizer, gs.loss], feed_dict=feed_dict_tr)
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
