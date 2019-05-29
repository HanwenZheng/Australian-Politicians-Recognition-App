"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : To test retrained Alexnet
 Modified based on Frederik Kratzert's code
 https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib
from alexnet import AlexNet

detector = dlib.get_frontal_face_detector()

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

image_dir = 'D:/Pycharm/projects/faces/test/KA'

# get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# load all images
imgs = []
for f in img_files:
    imgs.append(cv2.imread(f))

# plot images
fig = plt.figure(figsize=(15, 6))
for i, image2 in enumerate(imgs):
    fig.add_subplot(1, 6, i + 1)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.axis('off')

class_names = ('JA','JF','KA','KA2','MM','SM','TA')

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 7, [])

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax
softmax = tf.nn.softmax(score)
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    saver.restore(sess, 'checkpoint/model_epoch10.ckpt')

    # Create figure handle
    fig2 = plt.figure(figsize=(15, 6))

    # Loop over all images
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

        # Convert image to float32 and resize to (227x227)
        image2 = cv2.resize(image2.astype(np.float32), (227, 227))

        # Subtract the ImageNet mean
        image2 -= imagenet_mean

        # Reshape as needed to feed into model
        image2 = image2.reshape((1, 227, 227, 3))

        # Run the session and calculate the class probability
        probs = sess.run(softmax, feed_dict={x: image2, keep_prob: 1})

        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]

        # Plot image with class name and prob in the title
        fig2.add_subplot(1, 6, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Class: " + class_name + ", probability: %.4f" % probs[0, np.argmax(probs)])
        plt.axis('off')

    fig2.show()