"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : Print training images' path+lable to text files
"""

import glob
import os
import numpy as np
import fileinput

path = 'D:/Pycharm/projects/faces/tfaces'

def read_img(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    paths = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading image: %s' % (im))
            paths.append(im)
            labels.append(idx) #os.path.basename(os.path.dirname(im))
    return np.asarray(paths), np.asarray(labels, np.int32)
impath, label = read_img(path)

arr = np.arange(impath.shape[0])
np.random.shuffle(arr)
impath = impath[arr]
label = label[arr]

ratio = 0.8
s = np.int(len(impath) * ratio)
x_train = impath[:s]
y_train = label[:s]
x_val = impath[s:]
y_val = label[s:]

text_file = open("trainpath.txt", "w")
for x,y in zip(x_train, y_train):
    text_file.write(x + " " + str(y)+"\n")
text_file.close()
text_file = open("valpath.txt", "w")
for x,y in zip(x_val, y_val):
    text_file.write(x + " " + str(y)+"\n")
text_file.close()

with fileinput.FileInput("trainpath.txt", inplace=True) as file:
    for line in file:
        print(line.replace("\\", "/"), end='')
with fileinput.FileInput("valpath.txt", inplace=True) as file:
    for line in file:
        print(line.replace("\\", "/"), end='')

print("finish!")
