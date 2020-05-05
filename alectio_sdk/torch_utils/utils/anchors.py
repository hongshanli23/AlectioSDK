import numpy as np
import torch
import torch.nn as nn

class Anchors:
    '''Generate anchors box priors on one image
    a list of anchor dimensions
    
    Paramters:
    ----------
        dims: list of tuples. [(w0, h0), (w1, h1), ...]
            Each (wi, hi) represents the width and height 
            of 
        stride: how many pixels to skip 
        
    
    '''
    def __init__(self, stride, dims, img_dims, x_stride=):
        self.stride = stride
        self.dims = dims
        
        self.data = self._generate(img_dims)
        
    
    def normalize(self, convention='xyxy'):
        '''
        Paramters:
        ----------
            convention: {'xyxy', 'cxcy'}
                if 'xyxy' the returned normalized priors are 
        '''
        
        if convention=='xyxy':
            return 
        elif convention=='cxcy'
            pass
        else:
            raise ValueError(
                "convntion argument only takes on value 'xyxy' or 'cxcy',\
            got {}".format(convention))
            

    def _generate(self, img_dim):
        '''
        Paramte
        img_dim: tupe
        '''
        
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        
        # generate anchors at one focal point
        num_anchors = len(self.dims)
        
        anchors = np.zeros((num_anchors, 4))
        anchors[:,2:] = np.array(self.dims)

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:,0::2] -= np.tile(anchors[:,2]*0.5, (2,1)).T
        anchors[:,1::2] -= np.tile(anchors[:,3]*0.5, (2,1)).T       
        # proprogate the anchors to all focal points
        
        # generate meshgrid of focal points
        sx = np.arange(0, img_shape[0], self.x_stride) + 0.5*self.x_stride
        sy = np.arange(0, img_shape[1], self.y_stride) + 0.5*self.y_stride
        
        sx, sy = np.meshgrid(sx, sy)

        focal_points = np.vstack((sx.ravel(), sy.ravel(),
                sx.ravel(), sy.ravel())).transpose()

        A = anchors.shape[0]
        K = focal_points.shape[0]
        anchors = anchors.reshape((1, A, 4))
        
        focal_points = focal_points.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = (anchors + focal_points).reshape((K*A, 4))
     
        return torch.from_numpy(all_anchors).float()



