import os
import sys
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CMNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.align = glob(os.path.join(root, 'align', "*", "*"))
        self.conflict = glob(os.path.join(root, 'conflict', "*", "*"))

        # apply a balanced sampling strategy
        debias = True
        if debias:
            # balance the align num (decrease)
            self.align = self.align[0::4]
            # balance the conflict num (increase)
            self.conflict = self.conflict * 5 * 9
            
        self.data = self.align + self.conflict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        attr = torch.tensor([int(img_file.split('_')[-2]), int(img_file.split('_')[-1].split('.')[0])])
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)
        return img, attr

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root
        self.img_names = os.listdir(root)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_file = os.path.join(self.root, img_name)
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name


def get_data_loader(data_type='train'):
    batch_size = 64
    num_workers = 8 if sys.platform == 'linux' else 0

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    if data_type == 'train':
        dataset = CMNISTDataset('dataset/train', transform)
    else:
        dataset = TestDataset('dataset/test', transform)
    
    data_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True)
    return data_loader

