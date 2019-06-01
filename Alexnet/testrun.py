"""
 AUTHOR  : Hanwen Zheng, Wenshuang Cao, Zhiyuan Chen
 PURPOSE : Verify model accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet

image_dir = 'D:/Pycharm/projects/faces/test/TA'
checkpoint_path = 'checkpoint/model_epoch40.ckpt'

path2p = "deploy.prototxt.txt"
path2m = "res10_300x300_ssd_iter_140000.caffemodel"
confidencelvl = .5
net = cv2.dnn.readNetFromCaffe(path2p, path2m)

num_class = 53
class_names = ('Amanda_Stoker','Anne_Ruston','Anne_Urquhart','Anthony_Chisholm','Arthur_Sinodinos','Bill_Shorten','Bridget_McKenzie','Carol_Brown','Catryna_Bilyk','Chris_Ketter','Cory_bernardi','Dean_Smith','Deborah_O_Neill','Don_Farrell','Doug_Cameron','Eric_Abetz','James_Paterson','Jane_Hume','Jenny_McAllister','Jim_Molan','John_Alexander','Jonathon_Duniam','Josh_Frydenberg','Karen_Andrews','Kevin_Andrews','Kimberley_Kitching','Kim_Carr','Kristina_Keneally','Linda_Reynolds','Lisa_Singh','Louise_Pratt','Malarndirri_McCarthy','Marise_Payne','Mathias_Cormann','Matthew_Canavan','Michaelia_Cash','Michael_McCormack','Mitch_Fifield','Murray_Watt','Patrick_Dodson','Pauline_Hanson','Penny_Wong','Raff_Ciccone','Rex_Patrick','Richard_Colbeck','Richard_Di_Natale','Scott_Morrison','Simon_Birmingham','Stirling_Griff','Sue_Lines','Tanya_Plibersek','Tony_Abbott','Zed_Seselja')

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
imgs = []

for f in img_files:
    imgs.append(cv2.imread(f))

fig = plt.figure(figsize=(15, 6))
for i, image2 in enumerate(imgs):
    fig.add_subplot(1, 6, i + 1)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.axis('off')

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_class, [])
score = model.fc8
softmax = tf.nn.softmax(score)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)

    fig2 = plt.figure(figsize=(15, 6))

    for i, image in enumerate(imgs):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        innerConfidence = 0
        innerImage = image
        for j in range(0, detections.shape[2]):
            confidence = detections[0, 0, j, 2]
            if innerConfidence <= confidence:
                innerConfidence = confidence
            else:
                continue

            if confidence > confidencelvl:
                image2 = image
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
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
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                innerImage = image2

        if innerConfidence < confidencelvl:
            print('no face detected in image #'+(i+1))

        image3 = innerImage

        innerImage = cv2.resize(innerImage.astype(np.float32), (227, 227))
        innerImage -= imagenet_mean
        innerImage = innerImage.reshape((1, 227, 227, 3))

        probs = sess.run(softmax, feed_dict={x: innerImage, keep_prob: 1})
        class_name = class_names[np.argmax(probs)]
        fig2.add_subplot(1, 6, i + 1)
        plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
        plt.title(class_name + ": %.4f%%" % probs[0, np.argmax(probs)])
        plt.axis('off')

    fig2.show()
    
