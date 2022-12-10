# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    # def __init__(self, root, co_transform=None, subset='train'):
    def __init__(self, root, co_transform=None, subset = "val"):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        # print("self.images_root",type(subset))
        self.images_root += "val"
        self.labels_root += "val"

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
          
        # self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        # self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()
        posi2 = 0
        for f in self.filenamesGt:
            self.filenamesGt[posi2] = f +".png"
            posi2 += 1      

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        width = 320
        height = 320 
        dim = (width, height)

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
            # Only thing added to the original  
            image = image.resize(dim)

        #  original just keep same as the image and remove line 110 
        mask = os.path.join(self.labels_root, filenameGt)
        mask_cv = cv2.imread(mask, cv2.IMREAD_UNCHANGED)[:,:,0]
        mask_cv_r = cv2.resize(mask_cv, dim, interpolation = cv2.INTER_AREA).astype(int)

        return np.asarray(image)/255, np.asarray(mask_cv_r),filename,filenameGt

    def __len__(self):
        return len(self.filenames)    


# dataset_train = cityscapes('/home/preetham/Project_03_2Sem_data/cityscapes', 'train') #co_transform,
# for i in dataset_train:
#     print(i[0].size(), i[1].size())
# loader = DataLoader(dataset_train, num_workers=1, batch_size=1, shuffle=True)