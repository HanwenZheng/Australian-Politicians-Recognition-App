"""
 AUTHOR  : Hanwen Zheng, Zhiyuan Chen, Wenshuang Cao
 PURPOSE : To test retrained Alexnet
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib
from alexnet import AlexNet

detector = dlib.get_frontal_face_detector()
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

image_dir = 'D:/Pycharm/projects/faces/test/KA'
class_names = ('JA','JF','KA','KA2','MM','SM','TA')

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

model = AlexNet(x, keep_prob, 7, [])
score = model.fc8

#create op to calculate softmax
softmax = tf.nn.softmax(score)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'checkpoint/model_epoch10.ckpt')
   
    fig2 = plt.figure(figsize=(15, 6))
    for i, image in enumerate(imgs):

        dets = detector(image, 1)
        if len(dets) == 1:
            for index, face in enumerate(dets):
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if right > image.shape[1]:
                    right = image.shape[1]
                if bottom > image.shape[0]:
                    bottom = image.shape[0]
                # io.imsave('temp.jpg', img)
                image2 = image[top:bottom, left:right]
        else:
            if len(dets) == 0:
                print('ZERO face:')
            else:
                if len(dets) > 1:
                    print('MUTIPLE faces:')

        image2 = cv2.resize(image2.astype(np.float32), (227, 227))
        image2 -= imagenet_mean
        image2 = image2.reshape((1, 227, 227, 3))
        
        probs = sess.run(softmax, feed_dict={x: image2, keep_prob: 1})
        class_name = class_names[np.argmax(probs)]

        fig2.add_subplot(1, 6, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Class: " + class_name + ", probability: %.4f" % probs[0, np.argmax(probs)])
        plt.axis('off')

    fig2.show()
