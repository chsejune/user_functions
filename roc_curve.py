__author__ = 'Sejune Cheon'
__version__ = '1.0'

#dev. in python3

import numpy as np
from sklearn import metrics as mt
import matplotlib.pyplot as plt


def roc_curve_using_tfpn(tp, tn, fp, fn):
    ## data
    # tp = 18556
    # tn = 33
    # fp = 0
    # fn = 4

    y_pos_real = np.ones(tp+fn)
    y_neg_real = np.empty(tn+fp)
    y_neg_real.fill(-1)


    y_predict_postive_real_postive = np.ones(tp)
    y_predict_postive_real_negative = np.ones(fp)
    y_predict_negative_real_postive = np.empty(fn)
    y_predict_negative_real_postive.fill(-1)
    y_predict_negative_real_negative = np.empty(tn)
    y_predict_negative_real_negative.fill(-1)


    y_true = np.hstack((y_pos_real, y_neg_real))
    y_predict = np.hstack((y_predict_postive_real_postive, y_predict_negative_real_postive, y_predict_postive_real_negative, y_predict_negative_real_negative))

    fpr, tpr, thresholds = mt.roc_curve(y_true, y_predict, pos_label=1)
    area = mt.roc_auc_score(y_true, y_predict)

    plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area)
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 0, 1], [0, tpr[0], 1], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.plot([0,0 ], [0.472)
    plt.xlim([-0.007, 1.0])
    plt.ylim([0.0, 1.007])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()