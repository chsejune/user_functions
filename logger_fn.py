__author__ = 'Sejune Cheon'
__version__ = '1.0'
__environment__ = 'Python-3.5.2'

# tested on python 3.5

import logging

def set_logger(save_path):

    ## log file 기록을 위한 셋팅
    logger = logging.getLogger()

    ## 로그 포멧 지정
    # formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(message)s')

    ## 로그 출력 핸들러 셋팅
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream) # 콘솔창 출력을 위해

    # 학습 과정 로그 파일 기록을 위한 셋팅
    handler_file = logging.FileHandler(save_path)
    handler_file.setFormatter(formatter) # 기록 형식 지정
    logger.addHandler(handler_file) # 파일 기록을 위해

    logger.setLevel(logging.DEBUG) # log 기록 레벨

    return logger