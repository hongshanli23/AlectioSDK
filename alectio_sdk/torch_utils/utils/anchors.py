import numpy as np
import torch
from ._bbox import xy_to_cxcy



class Anchors:
    '''Generate anchors box priors on one image
    a list of anchor dimensions
    
    Paramters:
    ----------
        
        configs: list of dictionaries. Each dictionary
            specifies the definition of one type of anchor
            prior over the image. The dictionary looks like
            {
                "width": <width of the anchor in absolute pixel (int) 
                "height": <height of the anchor in absolute pixel (int)
                "x_stride": <stride along x-axis (int)
                    an anchor is generated along x-axis
                    for every x_stride pixels.
                    
                    top_x, top_y of the anchor is used as 
                    the point-of-definition of each anchor. 
                    The x-coordinate of the point-of-definitions
                    is computed as 
                        np.arange(0, image_width, x_stride)
                    >,
                    
            
                "y_stride": <stride along y-axis (int),
                    an anchor is generated along y-axis
                    for every y_stride pixels/
                    
                    top_x, top_y of the anchor is used as 
                    the point-of-definition of each anchor.
                    The y-coordinate of the point-of-definitions
                    is computed as:
                        np.arange(0, image_height, y_stride)
                        >
                    
            
            }
            
        image_width: width of the image (int).
        image_height: height of the image (int)
        
    Attributes:
    -----------
        data: the anchor priors in absolute pixels
    '''
    
    
    def __init__(self, configs, image_width, image_height):
        self.configs = configs
        self.image_width, self.image_height = image_width, image_height    
        self.data = self._generate()
        
    
    def normalize(self, convention='xyxy'):
        '''
        Return normalized anchor priors
        
        Paramters:
        ----------
            convention: {'xyxy', 'cxcy'}
                if 'xyxy' the returned normalized priors are 
                
        Return: 
        -------
            A tensor of shape (M, 4).
            M is the total number of anchors generated according
            to the self.configs. Convention of the returned anchors
            is determined by @convention
        '''
        if convention not in set(['xyxy', 'cxcy']):
            raise ValueError(
                "convntion argument only takes on value 'xyxy' or 'cxcy',\
            got {}".format(convention))
            
        # normalized data
        ndata = self.data.clone()
        ndata[:,0::2] /= float(self.image_width)
        ndata[:,1::2] /= float(self.image_height)
        
        if convention=='xyxy':
            return ndata
        elif convention=='cxcy':
            return xy_to_cxcy(ndata)
        

    def _generate(self):
        '''Generate anchor priors'''
        
        all_anchors = []
        for config in self.configs:
            # compute point-of-defintion of anchors
            x_stride, y_stride = config['x_stride'], config['y_stride']
            
            sx = np.arange(0, self.image_width, x_stride)
            sy = np.arange(0, self.image_height, y_stride)
            sx, sy = np.meshgrid(sx, sy)
            
            # point of definition for each anchor
            pod = np.stack([sx.ravel(), sy.ravel(), 
                           sx.ravel(), sy.ravel()]).transpose()
            
            # generate batch of anchor at the 0-th point-of-defition
            anchors = np.zeros((pod.shape[0], 4))
            w, h = config['width'], config['height']
            anchors[:,2]=w; anchors[:,3]=h
            
            # shift according to the point-of-definition
            anchors = pod + anchors
            all_anchors.append(anchors)
        
        all_anchors = np.concatenate(all_anchors, axis=0)
        
        return torch.from_numpy(all_anchors).float()
            
        



