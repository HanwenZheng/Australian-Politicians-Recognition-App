"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : Random crop
"""

import glob
import os
import cv2
from skimage import io
from random import random

percent2rc = 0.5
num2copy = 3
path = 'D:/Pycharm/projects/alexnet/faces/test/'
cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]

for idx, folder in enumerate(cate):
    # if idx == 0:
    for im in glob.glob(folder + '/*.jpg'):
        r = random()
        if r > percent2rc:
            print('Not random_cropping image: %s' % (im))
        else:
            print('Random_cropping image: %s' % (im))
            for x in range(num2copy):
                image = cv2.imread(im)
                (h, w) = image.shape[:2]
                percent = 0.9
                ch = h * percent
                cw = w * percent
                startPercent = 1 - percent
                h *= startPercent
                w *= startPercent
                r = random()
                startX = r * w
                r = random()
                startY = r * h
                image = image[int(startY):int(startY + ch), int(startX):int(startX + cw)]
                # image = cv2.resize(image, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                io.imsave(im.strip('.jpg') + '_randomCrop' + str(x) + '.jpg', image)
