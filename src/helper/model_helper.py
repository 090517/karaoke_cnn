# Imports
import numpy as np
import os.path as path
from os import listdir

class training_loader(object):
    def __init__(self,training_path, batch_size):
        self.path = training_path
        self.batch_size = batch_size
        path_content = listdir(training_path)
        self.training_files = [f for f in path_content 
                               if (path.isfile(path.join(training_path, f)) and 
                                   f.startswith("training_set"))]
        self.training_files.sort(key=lambda x : int(x.split(".")[0][12:]))
    
    def __iter__(self):
        for f in self.training_files:
            chunk_data = np.load(path.join(self.path, f))
            np.random.shuffle(chunk_data)
            amount_samples = chunk_data.shape[0]
            for chunk_index in range(0, amount_samples, self.batch_size):
                end_index = min(amount_samples, chunk_index + self.batch_size)
                yield (chunk_data["joined"][chunk_index:end_index],
                      chunk_data["truth"][chunk_index:end_index])