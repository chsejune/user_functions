__author__ = 'Sejune Cheon'
__version__ = '1.0'
__environment__ = 'Python-3.5.2'
__description__ = 'developed for preprocessing feature to image-wise data-form'

import numpy as np

def preprocess_feature_input(output_features):

    feature_images = []
    for img in output_features:
        feature_img = []
        for val in img.flatten():
            feature_img.append(np.full((150, 150), val))
        feature_images.append(np.rollaxis(np.array(feature_img), 0, 3))

    return np.array(feature_images)
