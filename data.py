import os
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data(TRAIN_DATA_PATH):
    images, labels, class_names = read_data(TRAIN_DATA_PATH)
    (train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    return [train_images, train_labels, test_images, test_labels, class_names]

def read_data(TRAIN_DATA_PATH):
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
                    ret_images.append(image)
                    ret_labels.append(float_count)

    sesh = tf.Session()
    for i in range(0, len(ret_images),500):
        ret_images[i:i+500] = tf.image.resize_images(np.array(ret_images[i:i+500]),[50,50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).eval(session=sesh)
        # ret_images[i:i+500] = tf.image.rgb_to_grayscale(np.array(ret_images[i:i+500])).eval(session=sesh)

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
