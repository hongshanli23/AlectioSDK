'''
test torch_utils functions
'''

from alectio_sdk.torch_utils.loss import HardNegativeMultiBoxesLoss
from alectio_sdk.torch_utils.metrics import mAP
from alectio_sdk.torch_utils.utils import Anchors
from alectio_sdk.torch_utils.utils import cxcy_to_xy

from numpy.testing import assert_almost_equal
import numpy as np

import torch
import torch.nn.functional as F


def test_anchors():
    configs = [
        {
            "width": 10,
            "height": 15,
            "x_stride" : 8,
            "y_stride" : 15
            
        }
        
    ]
    
    anchors = Anchors(configs, image_width=416, image_height=416)

    
    assert anchors.data.numpy().ndim == 2
    
    # expect number of anchors
    nx = np.arange(0, 416, configs[0]['x_stride']).shape[0]
    ny = np.arange(0, 416, configs[0]['y_stride']).shape[0]
    
    assert anchors.data.shape[0] == nx*ny
    assert anchors.data.shape[1] == 4
    
    xyndata, cxcyndata = anchors.normalize(), anchors.normalize('cxcy')
    
    assert_almost_equal(xyndata.numpy(), cxcy_to_xy(cxcyndata).numpy())
    


def test_hardnegative_multiboxesloss():
    #the loss should be zero when prediction matches the ground-truth
    
    # on a 416 x 416 image generate a 7 x 7 grid
    # generate anchors with dimension 
    # 10,13,  16,30,  33,23
    # use the anchors as the prior 
    
    x_stride, y_stride = 416 // 7, 416//7
    
    configs = [
        {
            "width" : 10,
            "height" : 13,
            "x_stride" : x_stride,
            "y_stride" : y_stride
        },
        {
            "width" : 16,
            "height" : 30,
            "x_stride" : x_stride,
            "y_stride" : y_stride
        },
        {
            "width" : 33,
            "height" : 27,
            "x_stride" : x_stride,
            "y_stride" : y_stride
        },
        
    ]
    
    anchors = Anchors(configs, 416, 416)
    
    priors = anchors.normalize('xyxy')

    
    loss_fn = HardNegativeMultiBoxesLoss(priors, device='cpu')
    
    n_priors = priors.shape[0]
    
    predicted_priors = torch.rand(1, n_priors, 4)
    predicted_objectness = F.softmax(torch.rand(1, n_priors), dim=1)
    predicted_class_dist = F.softmax(torch.rand(1, n_priors, 80), dim=2)
    
    boxes = [F.softmax(torch.rand(5, 4), dim=1)]
    labels = [torch.LongTensor(range(5))]
    
    loss_fn(predicted_priors, predicted_class_dist,
            predicted_objectness, 
            boxes, labels)
    
    
def test_mAP():
    # mAP should be 1 for perfect detection
    
    det_boxes = [
        cxcy_to_xy(F.softmax(torch.rand(10, 4), dim=1))
    ]
    
    det_labels = [
        torch.LongTensor(range(10))
        
    ]
    
    det_scores = [
        torch.ones(10, dtype=torch.float)
    ]
    
    true_boxes = det_boxes
    true_labels = det_labels
    
    true_difficulties =[
         torch.zeros(10, dtype=torch.float)
    ]
    
    n_classes = 10
    
    threshold=0.5
    
    ap_dict, map = mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,
                true_difficulties, n_classes, 'cpu', 
                 threshold)
    
    for c, v in ap_dict.items():
        assert v == 1.0
        
    assert map == 1.0
    
    

    
    
    
    
    
    