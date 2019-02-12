import os
import numpy as np
import math
import matplotlib.pyplot as plt


import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

import pickle

# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
evaluation = tf.placeholder_with_default(True, shape=())

# build network
net = ColorHandPose3DNetwork()
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# initialize network
net.init(sess)

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
            if letter == "A" or letter == "B":
                if len(image_path_list) > 0:
                    # iterate each image
                    float_count += 0.1
                    float_count = round(float_count,2)
                    ret_class_names.append(letter)
                    # print(letter, float_count)
                    # print("training set of "+str(len(image_path_list)))
                    # for i in range(len(image_path_list)):
                    #     # print(i)
                    #     # read image and get hand from image
                    #     image = get_hand(image_path_list[i])
                    #     #
                    #     # # add hand, letter to ret array
                    #     ret_images.append(image)
                    #     ret_labels.append(float_count)

    # with open("image_out","wb") as f:
    #     pickle.dump(ret_images,f)
    # with open("label_out","wb") as f:
    #     pickle.dump(ret_labels,f)
    ret_images = pickle.load( open("image_out","rb") )
    ret_labels = pickle.load( open("label_out","rb") )

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

def get_hand(image_path):
    image_raw = scipy.misc.imread(image_path)
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    # hand_scoremap_v = np.squeeze(hand_scoremap_v)
    # image_crop_v = np.squeeze(image_crop_v)
    # keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    return keypoint_coord3d_v

    # # post processing
    # image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    # coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    # coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
