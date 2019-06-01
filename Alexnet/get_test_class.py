"""
 AUTHOR  : Hanwen Zheng, Wenshuang Cao, Zhiyuan Chen
 PURPOSE : Supports testrun.py
"""

import os

path = 'D:/Pycharm/projects/All3'

cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]

for idx, folder in enumerate(cate):
    currentFolder = folder[folder.rfind('/') + 1:]
    print('\'' + currentFolder + '\',', end='')
