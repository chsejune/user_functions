__author__ = 'Sejune Cheon'
__version__ = '1.0'

#dev. in python3

import numpy as np

def random_sample_from_data(data, sampling_rate=0.1, sample_limit=20000):

    if data.shape[0] < sample_limit:
        sampling_rate = 1
    else:
        while data.shape[0] * sampling_rate < sample_limit:
            sampling_rate+=0.01
    # np.random.seed(1)
    return data[np.random.choice(data.shape[0], int(data.shape[0] * sampling_rate), replace=False), :]

