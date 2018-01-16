__author__ = 'Sejune Cheon'
__version__ = '1.0'
__environment__ = 'Python-3.5.2'

import scipy
import numpy as np


def imread(path, is_grayscale = True):
    """
    :param path: 불러들일 이미지 파일의 위치+파일명 입력
    :param is_grayscale: 이미지가 흑백인지 컬러인지 여부
    :return: numpy float 형태로 한장의 이미지를 출력해준다
    """
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)