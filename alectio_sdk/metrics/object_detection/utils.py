# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:50:47 2020

@author: arun
"""

import numpy as np


def compute_iou(detBoxes, GTBoxes):
    """
    Description:
        Computes IoU between Detected boxes and Ground truth boxes.
    Arguments:
        detBoxes : ndarray
            (N, 4) shaped array with detected bboxes
        GTBoxes : ndarray
            (M, 4) shaped array with Ground truth bboxes
        Returns: 
            (N, M) shaped numpy array with IoUs
    """
    area = (GTBoxes[:, 2] - GTBoxes[:, 0]) * (GTBoxes[:, 3] - GTBoxes[:, 1])

    iw = np.minimum(np.expand_dims(detBoxes[:, 2], axis=1), GTBoxes[:, 2]) - np.maximum(
        np.expand_dims(detBoxes[:, 0], 1), GTBoxes[:, 0]
    )
    ih = np.minimum(np.expand_dims(detBoxes[:, 3], axis=1), GTBoxes[:, 3]) - np.maximum(
        np.expand_dims(detBoxes[:, 1], 1), GTBoxes[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((detBoxes[:, 2] - detBoxes[:, 0]) * (detBoxes[:, 3] - detBoxes[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
    
    
def compute_ap(recall, precision):
    """ 
    Description:
        Compute the average precision, given the recall and precision curves.
    Arguments:
        recall:    The recall curve (list).
        precision: The precision curve (list).
    Returns:
        Average precision
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



