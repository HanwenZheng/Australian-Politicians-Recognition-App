"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : Mirror image
"""

import glob
import os
import cv2
from skimage import io
from random import random

percent2im = 1
path = 'D:/Pycharm/projects/faces/test/'
cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]

for idx, folder in enumerate(cate):
    if idx == 0:
        for im in glob.glob(folder + '/*.jpg'):
            r = random()
            if r > percent2im:
                print('Not image_mirroring image: %s' % (im))
            else:
                print('Image_mirroring image: %s' % (im))
                image = cv2.imread(im)
                image = cv2.flip(image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                io.imsave(im.strip('.jpg') + '_imageMirror.jpg', image)
