# !/usr/bin/env python

import sys 
import os

sys.path.append(os.path.abspath("/home/preetham/project_03_2Sem_code/ERFnet/erfnet_pytorch/"))
from eval import eval_cityscapes_color

def main():
    value = True
    i = 0
    while(value): 
        eval_cityscapes_color.main()
        # if i == 10:
        #     value = False
        # i += 1    
if __name__ == '__main__':
    main()    