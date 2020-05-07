import torch

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    boundary coordinates (xyxy-convention)
    
    Arguments:
    ----------
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
    --------
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in boundary coordinates.
    (xyxy-convention)
    
    Arguments:
    ----------
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
        
    Returns:
    --------
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# implementation from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates 
    (cx, cy, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    
    Paramters:
    ----------
        cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        
    Returns: 
    -------
        bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def batched_cxcy_to_xy(cxcy):
    '''Batched version of cxcy_to_xy
    
    Parameters:
    -----------
        cxcy: a tensor of shape (N, n_priors, 4)
            batched bboxes in cxcy format
    
    Return:
    -------
        batched bboxes in xyxy format
    '''
    return torch.cat([cxcy[:,:,:2] - (cxcy[:,:,2:]/2), 
                      cxcy[:,:,:2] + (cxcy[:,:,2:]/2)], dim=2)


# implementation from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) 
    to center-size coordinates (c_x, c_y, w, h).
    
    Paramters:
    ----------
       xy: bounding boxes in boundary coordinates (xyxy convention),
       a tensor of size (n_boxes, 4)
    
    Return:
    -------
         bounding boxes in center-size coordinates (cxcy convention), 
         a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


# implementation from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def batched_gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    '''Batched version of gcxgcy_to_cxcy
    
    Parameters:
    -----------
        gcxgcy: a tensor of shape (N, n_priors, 4)
            predicted boxes in batch in log space relative to 
            the anchor priors 
            
            N: the batch size
            n_priors: number of anchor priors on each image
            format of box is [gcx, gcy, pw, ph]
            
            gcx * w/10 + pcx = cx
            gcy * h/10 + pcy = cy
            pw * e^{gw/5} = w
            ph * e^{gh/5} = h
            
        
        priors_cxcy: a tensor of shape (N, n_priors, 4)
            pregenerated anchor priors in batch
            N: the batch size
            n_priors: number of priors
            each prior shoud follow cxcy format

    '''
    
    return torch.cat([
        gcxgcy[:,:,:2]*priors_cxcy[:,:,2:]/10 + priors_cxcy[:,:,:2],
        gcxgcy[:,:,2:]*torch.exp(priors_cxcy[:,:,2:]/5)], dim=2)

    

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) 
    w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, 
    and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    
    Parameters:
    -----------
        cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
        priors_cxcy: prior boxes with respect to which the encoding must be performed, 
            a tensor of size (n_priors, 4)
    
    
    Return: 
    -------
        encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h