"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : Gamma adjust
"""

import glob
import os
import numpy as np
import cv2
from skimage import io
from random import random

percent2ga = 1
num2copy = 3  # number 1 - 3
path = 'D:/Pycharm/projects/alexnet/faces/test/'
cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


for idx, folder in enumerate(cate):
    # if idx == 0:
    for im in glob.glob(folder + '/*.jpg'):
        r = random()
        if r > percent2ga:
            print('Not gamma_adjusting image: %s' % im)
        else:
            print('Gamma_adjusting image: %s' % im)
            image = cv2.imread(im)
            (h, w) = image.shape[:2]
            gamma = 0.9
            adjusted = adjust_gamma(image, gamma=gamma)
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            io.imsave(im.strip('.jpg') + '_adjustGamma' + str(0) + '.jpg', adjusted)
            if num2copy >= 2:
                gamma = 1.1
                adjusted = adjust_gamma(image, gamma=gamma)
                adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
                io.imsave(im.strip('.jpg') + '_adjustGamma' + str(1) + '.jpg', adjusted)
            if num2copy >= 3:
                gamma = 1.2
                adjusted = adjust_gamma(image, gamma=gamma)
                adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
                io.imsave(im.strip('.jpg') + '_adjustGamma' + str(2) + '.jpg', adjusted)
