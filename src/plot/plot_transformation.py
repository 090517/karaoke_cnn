import sys
import os
# Import workaround for parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt  
import numpy as np
import tensorflow as tf

from helper.plotting import *
from global_settings import *

plt.style.use('seaborn-deep')

index = 25

test_audio = np.load("../training_24k_center/training_set1000.npy")
train_spec = np.array(test_audio[index]["joined"])
truth_spec = np.array(test_audio[index]["truth"])

drop.training = True
with tf.Session(graph=model_graph, config=config) as sess:
    saver.restore(sess, "../model_data_24k_center/"+model_name)
    predicted_spec = sess.run(pred, feed_dict = {x: [train_spec]})

prediction = predicted_spec[0]
print("Input Shape:",train_spec.shape)
print("Truth Shape:",truth_spec.shape)
print("Prediction Shape:",predicted_spec[0].shape)

train_spec[train_spec<=0] = 1e-30
truth_spec[truth_spec<=0] = 1e-30
prediction[prediction<=0] = 1e-30

log_specs = [np.log10(train_spec[0:1000,4:21]), 
  			 np.log10(truth_spec[0:1000,:]),
  			 np.log10(prediction[0:1000,:])
			]

fig = plot_spec_sbs(log_specs,
					size = (10,6),
					aspect = 10,
					tick_scale = freq_res,
					vmin = -2)
fig.savefig("../diagrams/training_hor.png", format="png")
plt.show()

fig = plot_spec_sbs_vertical(log_specs,
					size = (6,10),
					aspect = 1/10,
					tick_scale = freq_res,
    				vmin=-2)

fig.savefig("../diagrams/training_vert.png", format="png")
# fig = plot_spec_sbs([train_spec[:,4:21], truth_spec, predicted_spec[0]], size = (17,8), aspect = 40)
plt.show()
