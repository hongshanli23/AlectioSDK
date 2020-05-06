from dataset import COCO, Transforms, collate_fn
from model import Darknet

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv

from alectio_sdk.torch_utils.loss import HardNegativeMultiBoxesLoss
from alectio_sdk.torch_utils.utils import Anchors
from alectio_sdk.torch_utils.metrics import mAP

device = 'cpu'

def train(payload):
    labeled = payload['labeled']
    
    batch_size = 16
    config_file = 'yolov3.cfg'
    
    coco = COCO('./data', Transforms(), samples=labeled, train=True)
    loader = DataLoader(coco, shuffle=True, batch_size=batch_size,
                       collate_fn=collate_fn)
    
    model = Darknet(config_file).to(device)
    
    
    optimizer = optim.Adam(model.parameters())
    # config the anchor priors
    
    image_width, image_height = 416, 416
    configs = [
        # anchors on 13 x 13 grid
        {
            'width': 116,
            'height': 90,
            'x_stride': image_width // 13,
            'y_stride': image_height // 13
            
        },
        {
            'width': 156,
            'height': 198,
            'x_stride': image_width // 13,
            'y_stride': image_height // 13
        },
        
        {
            'width': 373,
            'height': 326,
            'x_stride': image_width // 13,
            'y_stride': image_height // 13
            
        },
        
        # anchors on 26 x 26 grids
        {
            'width': 30,
            'height': 61,
            'x_stride': image_width // 26,
            'y_stride': image_height // 26
        },
        
        {
            'width': 62,
            'height': 45,
            'x_stride': image_width // 26,
            'y_stride': image_height // 26
            
        },
        {
            'width': 59,
            'height': 119,
            'x_stride': image_width // 26,
            'y_stride': image_height // 26
        },
        
        # anchors on 52 x 52 grid
        {
            'width': 10,
            'height': 13,
            'x_stride': image_width // 52,
            'y_stride': image_height // 52
        },
        
        {
            'width': 16,
            'height': 30,
            'x_stride': image_width // 52,
            'y_stride': image_height // 52
            
        },
        {
            'width': 33,
            'height': 23,
            'x_stride': image_width // 52,
            'y_stride': image_height // 52
        },
        
    ]
    
    # genrate anchors priors based on above config
    # anchors are normalized according to image dim
    # and they should follow xyxy convention
    anchors = Anchors(configs, image_width, image_height)
    priors = anchors.normalize('xyxy')

    loss_fn = HardNegativeMultiBoxesLoss(priors, device=device)
    
    model.train()
    for img, boxes, labels in loader:
        img = img.to(device)
        
        # 3 predictions from 3 yolo layers
        output = model(img)
        
        # batch predictions on each image
        batched_prediction = []
        for p in output: # (bacth_size, 3, gx, gy, 85)
            batch_size = p.shape[0]
            
            p = p.permute(1,2,3,0,4)
            p = p.view(-1, batch_size, 85)
            p = p.permute(1, 0, 2)
            
            batched_prediction.append(p)
            
        batched_prediction = torch.cat(batched_prediction, dim=1) # (batch_size, n_priors, 85)
        
        # the last dim of batched_prediction represent the predicted box
        # batched_prediction[...,:4] is the coordinate of the predicted bbox
        # batched_prediction[...,4] is the objectness score
        # batched_prediction[...,5:] is the pre-softmax class distribution
        
        # we need to apply some transforms to the those predictions
        # before we can use HardNegativeMultiBoxesLoss
        # In particular, the predicted bbox need to be relative to 
        # normalized anchor priors
        # we will define another function bbox_transform
        # to do those transform, since it will be used by other processes
        # as well. 
        # see documentation on HardNegativeMultiBoxesLoss
        # on its input parameters
        
        predicted_boxes, predicted_objectness, predicted_class_dist = bbox_transform(
                batched_prediction
        )
        
        loss = loss_fn(predicted_boxes, predicted_objectness, 
                       predicted_class_dist, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    return

def bbox_transform(batched_prediction):
    # apply sigmoid to bbox coordinate and objectness score
    batched_prediction[...,:5] = torch.sigmoid(batched_prediction[...,:5])
    
    predicted_boxes = batched_prediction[...,:4]
    predicted_objectness= batched_prediction[...,4]
    predicted_class_dist= batched_prediction[...,5:]
    return predicted_boxes, predicted_objectness, predicted_class_dist
    
    
    

if __name__ == '__main__':
    payload = {
        'labeled': [0, 1, 2, 3, 4]
    }
    train(payload)