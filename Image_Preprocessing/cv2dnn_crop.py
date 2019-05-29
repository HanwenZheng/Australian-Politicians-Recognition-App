"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : Batch face extraction
"""

import glob
import os
import numpy as np
import cv2
from skimage import io

confidencelvl = 0.5
path = 'D:/Pycharm/projects/faces/test/'
path2p = "deploy.prototxt.txt"
path2m = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(path2p, path2m)
cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]

for idx, folder in enumerate(cate):
    # if idx == 0:
    for im in glob.glob(folder + '/*.jpg'):
        print('reading image: %s' % (im))

        image = cv2.imread(im)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        innerConfidence = 0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if innerConfidence <= confidence:
                innerConfidence = confidence
            else:
                continue

            if confidence > confidencelvl:
                image2 = image
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX < 0:
                    startX = 0
                if startY < 0:
                    startY = 0
                if endX > image2.shape[1]:
                    endX = image2.shape[1]
                if endY > image2.shape[0]:
                    endY = image2.shape[0]
                if startX > endX or startY > endY:
                    innerConfidence = 0
                    continue
                distance = int((endY - startY) / 2 * 0.7)
                startX -= distance
                endX += distance
                startY -= distance
                endY += distance
                if startX < 0:
                    startX = 0
                if startY < 0:
                    startY = 0
                if endX > image2.shape[1]:
                    endX = image2.shape[1]
                if endY > image2.shape[0]:
                    endY = image2.shape[0]
                zuoyou = int((endX - startX) / 2)
                shangxia = int((endY - startY) / 2)
                midX = startX + zuoyou
                midY = startY + shangxia
                if shangxia < zuoyou:
                    startX = midX - shangxia
                    endX = midX + shangxia
                else:
                    startY = midY - zuoyou
                    endY = midY + zuoyou
                image2 = image2[startY:endY, startX:endX]
                # image2 = cv2.resize(image2, (224,224))
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                io.imsave(im.strip('.jpg') + '_crop.jpg', image2)

        if innerConfidence < confidencelvl:
            io.imsave(im.strip('.jpg') + '_REJECT.jpg', image)
            print(image + "REJECTED")
