import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import xy_to_cxcy, cxcy_to_xy
from ..utils import cxcy_to_gcxgcy, gcxgcy_to_cxcy
from ..utils import box_iou



class HardNegativeMultiBoxesLoss(nn.Module):
    '''Loss function for object detection
    
    This loss function computes the loss for bounding box regression and
    cross entropy loss for predicted objects. Loss for the background are 
    computed via hard-negative mining to avoid overwhelm the loss with background 
    class. 
    
    Parameters:
    -----------
        anchor_priors: Normalized bounding box in xyxy convention.
            prior anchor boxes on each image. Underlying data is a
            a tensor of shape (P, 4), where P denotes the total number
            of prior anchor boxes on each image. 
            The bounding boxes should follow xyxy convention
    
    Keyword Args:
    -------------
        neg_to_pos: int. Default 3
            number of negative priors to positive priors when computing loss.
            For example, if neg_to_pos is N, then for loss of m positive priors 
            (prior that contains an object) on each image, 
            we include loss of N*m hardest negative priors
        
        alpha: float. Default 1.0
            weights of bounding box regression error
        
        
    Shapes:
    -------
    predicted_boxes: tensor of shape (N, n, 4) (N: batch size, n: number of anchor boexes priors on each image)
        Predicted bounding boxes with respect to the anchor priors. 
        Let [px, py, pw, ph] be the predicted boxes relative to the anchor prior 
        [acx, acy, aw, ah] (in cxcy convention)
        Then the actual bounding box [px, py, pw, ph] represents is 
        [cx, cy, w, h] (in cxcy convention)
        
        cx = \frac{px}{10} \times aw + acx
        cy = \frac{py}{10} \times ah + acy
        w  = aw \times \exp{pw / 5}
        h  = ah \times \exp{ph / 5}
        
    predicted_objectness: a tensor of shape (N, n)
        N: batch size
        n: number of anchor boxes priors on each image
        Objectness of each predicted boxes. Range between
        0 and 1
        
        
    predicted_class_dist: a tensor of shape (N, n, m) (
        N: batch size, 
        n: number of anchor boexes priors on each image, 
        m: number of distinct classes (includeing the background class))
        Do not apply softmax
    
    
    boxes: a list of N tensors of shape (M_i, 4), $M_i$ is the number of 
        bounding boxes in each image. Each bbox should be normalized 
        by the spacial dimension of the image. 
        The bounding box should follow xyxy convention
    
    labels: class label for each image. A list of N tensors 
        [T1, T2, ...]
        Ti[j] corresponds to the object in image i box j
        Ti should be a torch.long tensor
    
    '''
    
    def __init__(self, anchor_priors, **kwargs):
        super(HardNegativeMultiBoxesLoss, self).__init__()
        
        self.anchor_priors = anchor_priors
        
        # number to negative priors to number of positive priors 
        self.neg_pos_ratio = kwargs.get('neg_pos_ratio', 3)
        
        # total loss = classification loss + alpha * regression loss 
        self.alpha = kwargs.get('alpha', 1.0)
        
        # level of iou for a detected bounding with ground-truth
        # bbox to be consider a positive detection
        self.threshold = kwargs.get('threshold', 0.5)
      
        self.device = kwargs.get('device')
        
    
    def forward(self, predicted_boxes, predicted_class_dist, 
                predicted_objectness,
                boxes, labels):
        
        batch_size = predicted_boxes.shape[0]
        
        # number of anchor boxes prior per image
        n_priors = self.anchor_priors.shape[0]
        
        # number of classes in the dataset
        n_classes=predicted_class_dist.shape[2]
        
        
        assert n_priors == predicted_boxes.shape[1] == predicted_class_dist.shape[1] == predicted_objectness.shape[1]
        
        # if n objects, then class label of background is n
        background_class = n_classes
        
        
        # true bboxes relative to the anchor priors
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, 
                                device=self.device)
        
        # true classes relative of each predicted_boxes
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long, 
                                  device=self.device)
        
        # compute loss for each image
        for i in range(batch_size):
            
            # number of non-background object on each image
            n_objects = boxes[i].shape[0]
            
            iou = box_iou(boxes[i], self.anchor_priors) #(num_objects, n_priors)
            
            # For each prior, find the object with max iou
            iou_for_each_prior, object_for_each_prior = iou.max(dim=0) # n_priors, n_priors
            
            # object_for_each_prior : [o_0, o_1, ..., o_{n_priors-1}] 
            # o_i is the object with highest iou to the prior i
            
            # We don't want a situation where an object is not represented in our postive 
            # (non-background) prior
            # 1 .An object might not have big iou with all priors and is therefore not in 
            # object_for each prior
            # 2. All priors with the object may be assgined as background based on threshold
            
            # To remedy this
            # First, find priors with the maximum iou for each object (mask)
            _, prior_for_each_object = iou.max(dim=1) 
            # prior_for_each_object: [x0, x1,...x_{n_objects-1}] xi is the index of the prior with highest iou with object i
            
            
            # Then assign each object to the corresponding prior with max iou
            object_for_each_prior[prior_for_each_object]=torch.LongTensor(
                range(n_objects), device=self.device)
            
            # object_for_each_prior = [o0, o1, o2,...o_{n_priors-1}]
            # oi is the local object (index of the box in bbox) with max iou with prior i
            # such that if an object z has the highest iou with prior j
            # then oj = z (note that a priori the prior oj might be hightest iou with objects other than z)
            
            
            # to make sure those prior selected above qualifies as positive priors
            # artificially give them iou > threshold
            iou_for_each_prior[prior_for_each_object] = 1.0
            
            
            # label for each prior 
            label_for_each_prior = labels[i][object_for_each_prior]
            # label_for_each_prior = [l_0, l_1, ..., l_{n_prior}]
            # such that l_j = labels[i][oj] (note oj is an index of the box in bboxes)
            
            # mask for the background
            background_mask = (iou_for_each_prior < self.threshold)
            
            label_for_each_prior[background_mask] = background_class
            
            # store the true labels for each priors
            true_classes[i] = label_for_each_prior
            
            # store the encoded ground-truth bbox relateive to the priors
            # in log space
            true_locs[i] = cxcy_to_gcxgcy(
                xy_to_cxcy(boxes[i][object_for_each_prior]), 
                xy_to_cxcy(self.anchor_priors)
            )
            
        # end for 
        positive_priors = true_classes != background_class # (N, num_priors)
        
        # localization loss is computed only over positive (non-background)
        # priors
        loc_loss = F.smooth_l1_loss(predicted_boxes[positive_priors], 
                                    true_locs[positive_priors])
        
        
        # classification loss on positive priors
        class_loss_pos = F.cross_entropy(
            predicted_class_dist[positive_priors].view(-1, n_classes),
            true_classes[positive_priors].view(-1) # positive priors has object
        
        )
        
        # objectness loss
        # classification loss is computeed over positive priors and most
        # difficult negative priors (hard negative mining)      
        # num positive and negative priors per image
    
   
        
        true_objectness = torch.zeros_like(predicted_objectness,
            device=self.device, dtype=torch.float)
        true_objectness[positive_priors] = 1.0
        
        obj_loss = F.mse_loss(predicted_objectness, true_objectness,
                reduction='none')
        
        # account loss for all postive class
        obj_loss_pos = obj_loss[positive_priors].mean()
        
        
        # hard mine negative
        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives
        obj_loss_neg = obj_loss.clone()
        
        # only consider negative loss
        obj_loss_neg[positive_priors] = 0.0
        
        # sort loss on each image
        obj_loss_neg, _ = obj_loss_neg.sort(dim=1, descending=True)
        obj_loss_hardneg = obj_loss_neg[:, :n_hard_negatives].mean()
        
        obj_loss = obj_loss_pos + obj_loss_hardneg
        
        return class_loss_pos + obj_loss + self.alpha * loc_loss
        
