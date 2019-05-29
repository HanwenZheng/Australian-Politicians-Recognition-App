"""
 AUTHOR  : Hanwen Zheng
 PURPOSE : To retain Alexnet
 Modified based on Frederik Kratzert's code
 https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""

# @author: Frederik Kratzert
"""Containes a helper class for image input pipelines in tensorflow."""

import numpy as np
import cv2
import dlib
from skimage import io

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes = 7):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.detector = dlib.get_frontal_face_detector()

        self.read_class_list(class_list)
        
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))
            
            #store total number of data
            self.data_size = len(self.labels)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            
            #flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            dets = self.detector(img, 1)
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
                    if right > img.shape[1]:
                        right = img.shape[1]
                    if bottom > img.shape[0]:
                        bottom = img.shape[0]
                    #io.imsave('temp.jpg', img)
                    img = img[top:bottom, left:right]
            else:
                if len(dets) == 0:
                    print('ZERO face: '+ paths[i])
                else:
                    if len(dets) > 1:
                        print('MUTIPLE faces: ' + paths[i])

            #rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)
            
            #subtract mean
            img -= self.mean
                                                                 
            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images, one_hot_labels
