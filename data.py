import os

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
    images, labels = read_data()
    #(train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    class_names = list("QWERTYUIOPASDFGHJKLZXCVBNM")
    exit()
    return [train_images, train_labels, test_images, test_labels, class_names]

# TODO
def read_data():
    for label in list(os.walk(TRAIN_DATA_PATH)):
        full_path, image_list = label[0], label[2]
        letter = full_path[len(TRAIN_DATA_PATH)+1:]
        image_path_list = [full_path+"/"+file for file in image_list]
        print letter, image_path_list
    return None, None

# TODO
def split_data(images, labels):
    pass
