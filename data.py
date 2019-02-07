import os
import numpy as np
import math
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = "../training_data"

def load_data():
    images, labels, class_names = read_data()
    (train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    return [train_images, train_labels, test_images, test_labels, class_names]

def read_data():
    """read data from files"""
    ret_images = []
    ret_labels = []
    ret_class_names = []
    float_count = 0.0 # arbitrary map from letter to float
    for label in list(os.walk(TRAIN_DATA_PATH)): # walk directory
        full_path, image_list = label[0], label[2]
        letter = full_path[len(TRAIN_DATA_PATH)+1:] # get letter class
        if len(letter) > 0:
            # get list of file paths to each image
            image_path_list = [TRAIN_DATA_PATH+"/"+letter+"/"+file for file in image_list]
            float_count += 0.1
            float_count = round(float_count,2)
            ret_class_names.append(letter)
            print(letter, float_count)
            if len(image_path_list) > 0:
                # iterate each image
                for i in range(len(image_path_list)):
                    # add image, letter to ret array
                    image = plt.imread(image_path_list[i])
                    # image = load_img(image_path_list[i])  # this is a PIL image
                    # image = img_to_array(image)
                    ret_images.append(image)
                    ret_labels.append(float_count)

    # sess = tf.Session()
    # for i in range(0, len(ret_images),10):
    #     ret_images[i:i+10] = tf.image.resize_images(np.array(ret_images[i:i+10]),[28,28]).eval(session=sess)
    #
    # ret_images = np.array(ret_images)
    # print(len(ret_images))
    # print(len(ret_images[0]))
    # print(type(ret_images))
    return np.array(ret_images), np.array(ret_labels), ret_class_names

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(images, labels):
    """split training and testing data"""
    train_percent = 0.7
    count = math.floor(len(images)*train_percent)
    images, labels = unison_shuffled_copies(images, labels)
    train_images, test_images = images[:count,:], images[count:,:]
    train_labels, test_labels = labels[:count], labels[count:]
    return (train_images, train_labels), (test_images, test_labels)
