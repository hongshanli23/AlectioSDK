from dataset import COCO, Transforms, collate_fn
from model import Darknet
import env

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv

import os

from alectio_sdk.torch_utils.loss import HardNegativeMultiBoxesLoss
from alectio_sdk.torch_utils.utils import Anchors, batched_gcxgcy_to_cxcy
from alectio_sdk.torch_utils.utils import batched_cxcy_to_xy


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# buiding anchor priors on one image
# the all image are resize to 416, 416
image_width, image_height = 416, 416

# config the dimension and stride of each 
# model of anchor 

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


# helper functions used to transform predictions
def bbox_transform(batched_prediction):
    # apply sigmoid to bbox coordinate and objectness score
    batched_prediction[...,:5] = torch.sigmoid(batched_prediction[...,:5])
    predicted_boxes = batched_prediction[...,:4]
    predicted_objectness= batched_prediction[...,4]
    predicted_class_dist= batched_prediction[...,5:]
    return predicted_boxes, predicted_objectness, predicted_class_dist
    


def train(payload):
    
    labeled = payload['labeled']
    resume_from = payload['resume_from']
    ckpt_file = payload['ckpt_file']
    
    # hyperparameters
    batch_size = 16
    epochs = 2 # just for demo
    lr = 1e-2
    weight_decay = 1e-2
   
    coco = COCO(env.DATA_DIR, Transforms(), samples=labeled, train=True)
    loader = DataLoader(coco, shuffle=True, batch_size=batch_size,
                   collate_fn=collate_fn)
    
    config_file = 'yolov3.cfg'
    model = Darknet(config_file).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # resume model and optimizer from previous loop
    if resume_from is not None:
        ckpt = torch.load(os.path.join(env.EXPT_DIR, resume_from))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # loss function
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
            p = p.view(batch_size, -1,  85)
            
            batched_prediction.append(p)
            
        batched_prediction = torch.cat(batched_prediction, dim=1) 
        # (batch_size, n_priors, 85)
        
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
    
    # save ckpt for this loop
    ckpt = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    
    torch.save(ckpt, os.path.join(env.EXPT_DIR, ckpt_file))
    return



def test(payload):
    ckpt_file = payload['ckpt_file']
    
    batch_size=16
     
    coco = COCO(env.DATA_DIR, Transforms(), train=False)
    loader = DataLoader(coco, shuffle=False, batch_size=batch_size,
                       collate_fn=collate_fn)
    
    config_file = 'yolov3.cfg'
    model = Darknet(config_file).to(device)
    
    ckpt = torch.load(os.path.join(env.EXPT_DIR, ckpt_file))
    model.load_state_dict(ckpt['model'])

    model.eval()
    
    # batch predictions from the entire test set
    predictions = []
   
    # keep track of ground-truth boxes and label 
    labels = []
    with torch.no_grad():
        for img, boxes, class_labels in loader:
            img = img.to(device)
            # get inference output
            output = model(img)
            
            for b, c in zip(boxes, class_labels):
                labels.append((b, c))
            
            # batch predictions from 3 yolo layers
            batched_prediction = []
            for p in output: # (bacth_size, 3, gx, gy, 85)
                p = p.view(p.shape[0], -1,  85)
                batched_prediction.append(p)
           
        
            batched_prediction = torch.cat(batched_prediction, dim=1) 
            predictions.append(batched_prediction)
            
    predictions = torch.cat(predictions, dim=0)
    
    
    # apply nms to predicted bounding boxes
    predicted_boxes, predicted_objectness, predicted_class_dist = bbox_transform(
        predictions
    )
    
    # the predicted boxes are in log space relative to the anchor priors
    # bring them back to normalized xyxy format
    cxcy_priors = anchors.normalize('cxcy')
    
    # expand the priors to match the dimension of predicted_boxes
    batched_cxcy_priors = cxcy_priors.unsqueeze(0).repeat(
        predicted_boxes.shape[0], 1, 1)
    
    predicted_boxes = batched_gcxgcy_to_cxcy(predicted_boxes, 
                                             batched_cxcy_priors)
    
    del batched_cxcy_priors 
    # convert predicted_boxes to xyxy format and perform nms
    xyxy = batched_cxcy_to_xy(predicted_boxes)
    del predicted_boxes # (no longer need cxcy format)
    
    # get predicted object
    # apply softmax to the predicted class distribution 
    # note that bbox_tranform does not apply softmax
    # because the loss we are using requires us to use raw output
    predicted_objects = torch.argmax(
        F.softmax(predicted_class_dist, dim=-1), dim=-1)

    # predictions on the test set (value of "predictions" of the return)
    prd = {}
    for i in range(len(coco)):
        # get boxes, scores, and objects on each image
        _xyxy, _scores = xyxy[i], predicted_objectness[i]
        _objects = predicted_objects[i]
        
        keep = tv.ops.nms(_xyxy, _scores, 0.5)
        boxes, scores, objects = _xyxy[keep], _scores[keep], _objects[keep]
        
        prd[i] = {
            "boxes" : boxes.cpu().numpy().tolist(),
            "objects" : objects.cpu().numpy().tolist(),
            "scores"  : scores.cpu().numpy().tolist()
            
        }
     
    # ground-truth of the test set
    # skip "difficulties" field, because every object in COCO 
    # should be considered reasonable
    lbs = {}
    for i in range(len(coco)):
        boxes, class_labels = labels[i]
        
        lbs[i] = {
            "boxes" : boxes.cpu().numpy().tolist(),
            "objects" : class_labels
        }
        
    return {"predictions": prd, "labels": lbs}
        
        
def infer(payload):
    unlabeled = payload['unlabeled']
    ckpt_file = payload['ckpt_file']
    
    batch_size=16
     
    coco = COCO(env.DATA_DIR, Transforms(), samples=unlabeled, train=True)
    loader = DataLoader(coco, shuffle=False, batch_size=batch_size,
                       collate_fn=collate_fn)
    
    config_file = 'yolov3.cfg'
    model = Darknet(config_file).to(device)
    ckpt = torch.load(os.path.join(env.EXPT_DIR, ckpt_file))
    model.load_state_dict(ckpt['model'])

    model.eval()
    
    # batch predictions from the entire test set
    predictions = []
  
    with torch.no_grad():
        for img, _, _ in loader:
            img = img.to(device)
            # get inference output
            output = model(img)
            
            # batch predictions from 3 yolo layers
            batched_prediction = []
            for p in output: # (bacth_size, 3, gx, gy, 85)
                batch_size = p.shape[0]
                p = p.view(batch_size, -1,  85)
                batched_prediction.append(p)

            batched_prediction = torch.cat(batched_prediction, dim=1) 
        predictions.append(batched_prediction)
    predictions = torch.cat(predictions, dim=0)
    
    
    # apply nms to predicted bounding boxes
    predicted_boxes, predicted_objectness, predicted_class_dist = bbox_transform(
        predictions
    )
    
    # the predicted boxes are in log space relative to the anchor priors
    # bring them back to normalized xyxy format
    cxcy_priors = anchors.normalize('cxcy')
    
    # expand the priors to match the dimension of predicted_boxes
    batched_cxcy_priors = cxcy_priors.unsqueeze(0).repeat(
        predicted_boxes.shape[0], 1, 1)
    
    predicted_boxes = batched_gcxgcy_to_cxcy(predicted_boxes, 
                                             batched_cxcy_priors)
    
    del batched_cxcy_priors
    
    # convert predicted_boxes to xyxy format and perform nms
    xyxy = batched_cxcy_to_xy(predicted_boxes)
    
    del predicted_boxes # (no longer need cxcy format)
    
 
    # class distribution is part of the return
    # do notapply softmax to the predicted class distribution 
    # as we will do it internally for efficiency
    outputs = {}
    for i in range(len(coco)):
        # get boxes, scores, and objects on each image
        _xyxy, _scores = xyxy[i], predicted_objectness[i]
        _pre_softmax = predicted_class_dist[i]
        
        keep = tv.ops.nms(_xyxy, _scores, 0.5)
        
        boxes, scores, pre_softmax = _xyxy[keep], _scores[keep], _pre_softmax[keep]
        
        outputs[i] = {
            "boxes" : boxes.cpu().numpy().tolist(),
            "pre_softmax" : pre_softmax.cpu().numpy().tolist(),
            "scores"  : scores.cpu().numpy().tolist()
            
        }
        
   
    return {"outputs" : outputs}
        
    

if __name__ == '__main__':
    # debug
    # train
    payload = {
        'labeled': range(10),
        'resume_from': None,
        'ckpt_file': 'ckpt_0'
    }
    train(payload)
    
    # test
    payload = {
        'ckpt_file': 'ckpt_0'
    }
    test(payload)
    
    # infer
    payload = {
        'unlabeled': range(10, 20),
        'ckpt_file': 'ckpt_0'
    }
    infer(payload)