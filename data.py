import os
import numpy as np
import math
from keras.preprocessing.image import array_to_img, img_to_array, load_img

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
                    image = load_img(image_path_list[i])  # this is a PIL image
                    image = img_to_array(image)
                    ret_images.append(image)
                    ret_labels.append(float_count)

    return np.array(ret_images), np.array(ret_labels), ret_class_names

def split_data(images, labels):
    """split training and testing data"""
    train_percent = 0.7
    count = math.floor(len(images)*train_percent)
    train_images, test_images = images[:count,:], images[count:,:]
    train_labels, test_labels = labels[:count], labels[count:]
    return (train_images, train_labels), (test_images, test_labels)
