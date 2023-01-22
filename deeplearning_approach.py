# !/usr/bin/env python
#!/usr/bin/python

# TOtal 4 parameter to set

import logging
import math
from random import randint, random
from tkinter import LEFT
from tracemalloc import stop
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image as im
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Twist
import sys
bridge = CvBridge()
from PIL import Image as I
from PIL import ImageDraw

# Import the local ML path
import sys 
import os
from subprocess import call
from numpy import asarray
from matplotlib.pyplot import imshow

#*************************************************************
# sys.path.append(os.path.abspath("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/"))
# from eval import *
# os.system("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/eval/eval_cityscapes_color.py datadir = '/home/preetham/Project_03_2Sem_data/gazebo_input' loadWeights = '../save/erfnet_training1/model_best.pth'")
# sys.path.append(os.path.abspath("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/eval/"))
# sys.path.append(os.path.abspath("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/"))
# from eval import dataset
# from eval import erfnet
# from eval import eval_cityscapes_color
# from eval import dataset
# from eval import
# from eval import
# from eval_cityscapes_color import *
# from pydoc import importfile
# module = importfile("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/eval/eval_cityscapes_color.py")
# ***********************************************

def image_callback(ros_image):
    # import ipdb; ipdb.set_trace()
    # sub = rospy.Subscriber("scan", LaserScan, scan_callback)
    # print("obstacle status",obstacle )
    # import ipdb; ipdb.set_trace()

# Read image from the ROS and save the image in the folder.
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    cv_image_path = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # ORIGINAL IMAGE OUTPUT
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    img = cv2.cvtColor(cv_image , cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image window", img)
    # cv2.waitKey(3)
    # cv2.show()

    # To save the image which will be used by the iterator.py to run the ML algo 
    path = '/home/preetham/Project_03_2Sem_data/gazebo_input/leftImg8bit/test'
    cv2.imwrite(os.path.join(path , 'img.jpg'), img)   

    # To read the image from the folder using the pil library
    # img_image = I.open('/home/preetham/Project_03_2Sem_data/gazebo_input/leftImg8bit/test/img.jpg')                # Note(Para1) - change location according to where you save
    # img_image = I.open(img) 
    # imshow(np.asarray(img_image))
    # plt.show()

#  CHECKING THE TRAVERSIBLITY 
#     # To read the segmented output from the ML which is run by the iterator.py
    seg_image = I.open('/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/eval/gazebo_output/test/img.jpg') # Note(Para2) - Change the path
    
#    # convert image to numpy array
    pred_mask_np = asarray(seg_image)
    imshow(np.asarray(pred_mask_np))
    plt.show()
#     # print("Image shape", pred_mask_np.shape)

#     # Initilize the window 
    init_window = np.arange(0,321,10)
    # print("INITIALIZE THE WINDOW",init_window)
    final_window = np.arange(0,321,10)

    # Robot size 
    robot_size = 110                                                                                                 # Tuneable(para 3) according to the robot size 
    image_size = pred_mask_np.shape[1]

    # final window start position 
    final_window_start_position = image_size - robot_size

    # Init_window
    for i in init_window:
        if i > final_window_start_position:
            final_window = np.delete(final_window, np.where(final_window == i))
    # print("FINAL WINDOW",final_window)

    # checking the traversibilty value
    traversible = np.array(())

    #  Plotting the traversible window 
    for window in final_window:
        pred_mask_np_each_window = pred_mask_np[:,window:window+robot_size]
        # print("pred_mask_np_each_window", window, "Window and robot ", window+robot_size)
        sum_each_window = np.sum(pred_mask_np_each_window)
        # print("sum_each_window",sum_each_window)
        traversible = np.append(traversible,sum_each_window)
        # print("traversivle",traversible)
    print("traversible",traversible)

    tick_label = final_window
    # plt.bar(final_window, traversible, tick_label = tick_label,
            # width = 1, color = ['red', 'green'])
    plt.bar(final_window, traversible, tick_label = tick_label,
            width = 100, color = [ 'green'])  
    plt.show()

    # traversible window 
    final_window_index = np.argmax(traversible)
    print("MOVE IN THE POSITION - ", np.argmax(traversible))   

    # TESTING IMAGE BASED ON UPLOADED IMAGE
    # Visualize the traversible part
    img_wind = I.open('/home/preetham/Project_03_2Sem_data/gazebo_input/leftImg8bit/test/img.jpg').resize((320, 320))
    # img_wind2 = Image.open('image2.png')
    # img_wind3 = Image.open('image2.png')

    width, height = img_wind.size
    top = 0
    bottom = height
    start = 0
    end = width

    # checking the traversiblity first time 
    draw = ImageDraw.Draw(img_wind)
    draw.rectangle((start, top, final_window[final_window_index], bottom), fill=(0, 192, 192), outline=(255, 255, 255))
    draw.rectangle((final_window[final_window_index]+robot_size, top, end , bottom), fill=(0, 192, 192), outline=(255, 255, 255))

    plt.imshow(np.asarray(img_wind))
    plt.show()

    # Publish the image 
    img_wind =np.asarray(img_wind)
    pub_windowimg = rospy.Publisher('windowimg', im,queue_size=1)
    # print("RIGHT LANE CHANGE FLAG ON STARIGHT NOW" )
    pub_windowimg.publish(bridge.cv2_to_imgmsg(img_wind))

def main(args):
    rospy.init_node('image_converter', anonymous=True)
    # for turtlebot3 waffle
    image_topic = "/camera/rgb/image_raw"
    # for usb cam
    # image_topic="/usb_cam/image_raw"                                                                       # para 4 change to usb cable
    image_sub = rospy.Subscriber(image_topic, im, image_callback)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1) 
    rate = rospy.Rate(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
