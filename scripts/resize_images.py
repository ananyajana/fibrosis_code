# we remove an image when the mean of the images falls below a thresold
# we determined the threshold to be 2. compared with the previous logic.
# the number of images removed were the same for Patient_1
import os
import shutil

basepath = '../data/lits_131_pat_all_train'

import math

import numpy as np
from PIL import Image
from skimage import io

str = 'Patient'
H = W = 224

for dirpath, dirs, files in os.walk(basepath): # recurse through the current directory
    for dirname in dirs:
        dname = os.path.join(dirpath,dirname)   # get the full direcotry name/path
        print(dirname)
        onlyfiles = [f for f in os.listdir(dname) if os.path.isfile(os.path.join(dname, f)) and '.png' in f]    # check files in a particular directory
        if len(onlyfiles) != 0:
            for i in range(len(onlyfiles)):
                img = dname + '/' + onlyfiles[i]
                im = Image.open(img)
                w, h = im.size
                #print('w {}, h {}'.format(w, h))
                #raise ValueError('Exit!!')
                new_size = (W, H)
                new_im = im.resize(new_size)
                #new_im.save(img)
                io.imsave(img, np.array(new_im))
                #print('{}, {}'.format(w, h))
