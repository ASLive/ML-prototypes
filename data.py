import os
import keras
import cv2

TRAIN_DATA_PATH = "../training_data"

def preprocess(func):
    def wrapper(*args, **kwargs):
        # load data
        ret = func(*args, **kwargs)
        # scale train/test images to 0-1 values
        ret[0] = ret[0] / 255.0
        ret[2] = ret[2] / 255.0
        return tuple(ret)
    return wrapper

@preprocess
def load_data():
    # images, labels = read_data()
    # (train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    # class_names = list("QWERTYUIOPASDFGHJKLZXCVBNM")
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return [train_images, train_labels, test_images, test_labels, class_names]

def read_data():
    ret_images = []
    ret_labels = []
    for label in list(os.walk(TRAIN_DATA_PATH)):
        full_path, image_list = label[0], label[2]
        letter = full_path[len(TRAIN_DATA_PATH)+1:]
        image_path_list = [full_path+"/"+file for file in image_list]
        if len(image_path_list) > 0:
            ret_images.append(cv2.imread(image_path_list[0]))
            ret_labels.append(letter)

    return ret_images, ret_labels

#TODO
def split_data(images, labels):
    train_images = images[:-1]
    train_labels = labels[:-1]
    test_images = [images[-1]]
    test_labels = [images[-1]]
    return (train_images, train_labels), (test_images, test_labels)
