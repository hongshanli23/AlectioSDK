'''
map dataset object for pytorch model
'''
from torch.utils.data import Dataset
import os
import PIL.Image as Image

class ImageDataCLS(Dataset)
    '''Image dataset object for classification'''
    def __init__(self, root, train=True, transform=None):
        '''
        root(str): root of data directory
        train (bool): use as train set
        transform(callable) transform applied to imgs
        '''
        self.transform=transform

        if train:
            imgdir=os.path.join(root, 'train')
            labels='train_labels.txt'
        else:
            imgdir=os.path.join(root, 'test')
            labels='test_labels.txt'

        # load images path and labels
        self.fpath=[]
        for im in os.listdir(imgdir):
            self.fpath.append(
                os.path.join(root, imgdir, im))
            
        self.target=[]
        with open(os.path.join(root, labels), 'r') as f:
            for ln in f.readlines():
                t = ln.split(',')[-1].strip()
                self.target.append(int(t))

    def __len___(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        img = Image.open(self.fpath[idx])
        t = self.target[idx]

        if self.transform:
            img = self.transform(img)
        return img, t




