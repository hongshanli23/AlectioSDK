from imagedata import ImageDataCLS
import torchvision.transforms as transforms

def test():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010))
        ])

    ds=ImageDataCLS(root='/home/ubuntu/DataLake/Data/CIFAR10DEBUG',
            transform=train_transform)
    
    for i in range(10):
        im, t = ds[i]
        print(im.shape, t)

test()
