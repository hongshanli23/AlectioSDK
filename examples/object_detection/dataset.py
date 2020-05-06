'''
parse coco data set as a pytorch Dataset object

'''
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
import PIL.Image as Image
import numpy as np

class COCO(Dataset):
    '''COCO dataset from Darknet format
    
    Bounding boxes are in xyxy format
    
    Paramters:
    ----------
    root: root directory to the dataset
    
    transform: callable
        tranform applied to both raw PIL label and list of bbox in xyxy format
    
    train: bool. Default True
        use for training
    
    '''
    def __init__(self, root, transform=None, samples=None, train=True):
        
        self.transform=transform
        
        if train:
            subdir='train2014'
        else:
            subdir='val2014'
        
        imgs = os.listdir(os.path.join(root, 'images', subdir))
        
        if samples:
            imgs = [imgs[i] for i in samples]
            
        self.image_paths = []
        for img in imgs:
            self.image_paths.append(os.path.join(root, 'images', subdir, img))
        
        # get label paths
        self.label_paths=[]
        for x in self.image_paths:
            self.label_paths.append(
                x.replace('images', 'labels').replace('.jpg', '.txt'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_label(self, path):
        '''load bbox on each image as a numpy array'''
        return np.loadtxt(path, dtype=np.float, delimiter=' ')
    
    def __getitem__(self, ix):
        img, label=Image.open(self.image_paths[ix]), self.load_label(
            self.label_paths[ix])
        
        if self.transform:
            return self.transform(img, label)
        else:
            return img, label


class Transforms(object):
    '''Transforms to apply to COCO image and its bounding bbox'''
    def __init__(self, img_size=416):
        # image transforms
        self.img_size = img_size        
        
    def __call__(self, img, label):
        img = self.img_transforms(img)
        
        # now label include bbox coordiate and the class label of object
        # split into bbox coordiate and class labels 
        # to be used later for loss function
        label = self.bbox_transforms(label)
        class_label = label[:,0].long()
        boxes = label[:,1:]
        
        return img, boxes, class_label
    
    def img_transforms(self, img):
        img = T.Resize((self.img_size, self.img_size))(img)
        img = T.ToTensor()(img)
        
        # stack gray scale image on top of itself 3 times to 
        # make it look like rgb image
        if img.shape[0]==1:
            img = torch.repeat_interleave(img, repeats=3, dim=0)
        
        return img.unsqueeze(0)
        
        
    def bbox_transforms(self, bbox):
        '''transofmr bbox from numpy array to torch.tensor
        each bbox will be converted to a np.array of shape (N, 5)
        '''
        assert isinstance(bbox, np.ndarray)
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        
        return torch.tensor(bbox, dtype=torch.float)
        

def collate_fn(batch):
    '''
    batch images along dim=0 (already unsqueezed by T.ToTensor).
    Put all boxes in a list
    Put all labels in a list
    '''
    
    images, boxes, labels = [], [], []
    for img, box, label in batch:
        images.append(img); boxes.append(box); labels.append(label)
    
    return torch.cat(images, dim=0), boxes, labels
    
    
        
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    coco = COCO('./data', transform=Transforms())
    
    loader = DataLoader(coco, batch_size=10, collate_fn=collate_fn)
    for img, label in loader:
        print(img.shape)
        
        
        
