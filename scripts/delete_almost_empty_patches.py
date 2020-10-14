# we remove an image when the mean of the images falls below a thresold
# we determined the threshold to be 2. compared with the previous logic.
# the number of images removed were the same for Patient_1
import os
import shutil

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--base_path', type=str, default='')


args = parser.parse_args()
basepath = args.base_path

import math

import numpy as np
from PIL import Image

str = 'Patient'
thresh = 2.02 # for full resolution images
#thresh = 5.0 # for resized images


imgs_removed = 0

for dirpath, dirs, files in os.walk(basepath): # recurse through the current directory
    for dirname in dirs:
        dname = os.path.join(dirpath,dirname)   # get the full direcotry name/path
        print(dirname)
        if str in dname:
            onlyfiles = [f for f in os.listdir(dname) if os.path.isfile(os.path.join(dname, f))]    # check files in a particular directory
            for i in range(len(onlyfiles)):
                img = dname + '/' + onlyfiles[i]
                im = Image.open(img)
                pix = np.array(im)
                cur_mean = pix.mean()
                if cur_mean < thresh:
                    os.remove(img)
                    imgs_removed += 1


