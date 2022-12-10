from os.path import join, isdir
from os import listdir, rmdir
from shutil import move, rmtree, make_archive

import os
import cv2
import glob
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

GT_DIR = '/home/preetham/Project_03_2Sem_data/cityscapes/gtFine/'
IMG_DIR = '/home/preetham/Project_03_2Sem_data/cityscapes/leftImg8bit/'

for parent in listdir(GT_DIR):
    parent_dir = GT_DIR + parent
    for child in listdir(parent_dir):
        if isdir(join(parent_dir, child)):
            keep = glob.glob(join(parent_dir, child) + '/*_gtFine_color.png')
            # keep = glob.glob(join(parent_dir, child) + '/*_gtFine_color')
            keep = [f.split('/')[-1] for f in keep]
            for filename in list(set(listdir(join(parent_dir, child))) & set(keep)):
                move(join(parent_dir, child, filename), join(parent_dir, filename))
            rmtree(join(parent_dir, child))

for parent in listdir(IMG_DIR):
    parent_dir = IMG_DIR + parent
    for child in listdir(parent_dir):
        if isdir(join(parent_dir, child)):
            for filename in listdir(join(parent_dir, child)):
                move(join(parent_dir, child, filename), join(parent_dir, filename))
            rmtree(join(parent_dir, child))


# # # # process anr archive image in smaller size
# IMG_SHAPE = (1024, 2048)

# gt_train_paths = [GT_DIR+'train/' + path for path in listdir(GT_DIR+'train/')]
# gt_test_paths = [GT_DIR+'test/' + path for path in listdir(GT_DIR+'test/')]
# gt_val_paths = [GT_DIR+'val/' + path for path in listdir(GT_DIR+'val/')]
# gt_paths = gt_train_paths + gt_test_paths + gt_val_paths

# im_train_paths = [IMG_DIR+'train/' + path for path in listdir(IMG_DIR+'train/')]
# im_test_paths = [IMG_DIR+'test/' + path for path in listdir(IMG_DIR+'test/')]
# im_val_paths = [IMG_DIR+'val/' + path for path in listdir(IMG_DIR+'val/')]
# im_paths = im_train_paths + im_test_paths + im_val_paths

# def resize_image(path):
#     img = Image.open(path)
#     img.thumbnail(IMG_SHAPE)
#     out_file = join(path)
#     img.save(out_file, 'PNG')

# for img in gt_paths + im_paths:
#     resize_image(img)  


