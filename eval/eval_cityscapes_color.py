# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

# 6 paramter to change

import numpy as np
import torch
import os
import importlib
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
# from dataset import cityscapes
# from erfnet import ERFNet
#  CHANGES FOR THE ROS or other to run the iterate from the other file
import sys
sys.path.append(os.path.abspath("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/eval"))      # Note tunable according to the 6 para
from dataset import cityscapes 
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 2

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform_cityscapes = Compose([
    Resize((512,1024),Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),  
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(255, 0),
    ToPILImage(),
])

def main():
    # modelpath = args.loadDir + args.loadModel
    modelpath = "/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/trained_models/"                             # 1 para
    # weightspath = args.loadDir + args.loadWeights
    weightspath = "/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/save/erfnet_training1/model_best.pth"      # 2 para

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #Import ERFNet model from the folder
    #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    model = ERFNet(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not 'store_true'):
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    # if(not os.path.exists(args.datadir)):
    #     print ("Error: datadir could not be loaded")

# Input Image Folder
    # loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
    # loader = DataLoader(cityscapes('/home/preetham/Project_03_2Sem_data/gazebo_input', input_transform_cityscapes, target_transform_cityscapes),
        # num_workers=4, batch_size=1, shuffle=False)                                         # 4 Para
    loader = DataLoader(cityscapes('/home/preetham/Project_03_2Sem_data/crop_traversible', input_transform_cityscapes, target_transform_cityscapes),
        num_workers=4, batch_size=1, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it.
    if ( not 'store_true'):
        vis = visdom.Visdom()

    # for step, (images, labels, filename, filenameGt) in enumerate(loader):
    for step, (images, filename) in enumerate(loader):
        if ( not 'store_true'):
            images = images.cuda()
            #labels = labels.cuda()

        inputs = Variable(images).type(torch.float).permute(0,3,1,2)
        # targets = Variable(labels).long()            

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # inputs, labels = inputs.to(device), labels.to(device)
        # inputs = inputs.cuda()
        # labels = labels.cuda()

        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        #label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        # label_color = Colorize()(label.unsqueeze(0))
        # label_color = 255 * label.unsqueeze(0)   # Commented to check the traversiblity
        # print(label_color)
        label_color = torch.where(label==0,255,0).type(torch.uint8)
        # print(label_color)
        # import ipdb; ipdb.set_trace()
        

# Output Image Folder 
        # filenameSave = "./crop_output/" + filename[0].split("leftImg8bit/")[1]
        # filenameSave = "./gazebo_New_output/" + filename[0].split("leftImg8bit/")[1]
        # filenameSave = "./New_output/" + filename[0].split("leftImg8bit/")[1]
        # filenameSave = "./del_output/" + filename[0].split("leftImg8bit/")[1]            # Para 5
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        #image_transform(label.byte()).save(filenameSave)      
        label_save = ToPILImage()(label_color)  
        # label_save.convert('L')        
        label_save.save(filenameSave) 

        if ( not 'store_true'):
            vis.image(label_color.numpy())
        print (step, filenameSave)

if __name__ == '__main__':
    main()    
