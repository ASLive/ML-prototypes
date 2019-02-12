# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import np_utils

from os.path import isfile

JSON_PATH = "./model.json"
WEIGHTS_PATH = "./model.h5"

def setup():

    return keras.Sequential([
        keras.layers.Flatten(input_shape=(75, 100, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)

    ## trying out a new ML model, runs with errors

    # return Sequential([
    #     #keras.layers.Flatten(input_shape=(75, 100)),
    #     #keras.layers.Dense(128, activation=tf.nn.relu),
    #     #keras.layers.Dense(10, activation=tf.nn.softmax),
    #
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(75, 100)),
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(10, activation='softmax')

        ])



def compile(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)

def ml_model(train_images, train_labels, retrain=False, save=True):
    if (not retrain) and isfile(JSON_PATH) and isfile(WEIGHTS_PATH):
        return read_model()
    else:
        return make_model(train_images, train_labels, save=save)

def make_model(train_images, train_labels, save=True):
    model = setup()
    compile(model)
    print(train_labels)
    model.fit(train_images, train_labels, epochs=5, batch_size=100) # train
    return save_model(model) if save else model

def save_model(model):
    # serialize model to json
    model_json = model.to_json()
    with open(JSON_PATH, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to hdf5
    model.save_weights(WEIGHTS_PATH)
    return model

def read_model():
    # load json and create model
    json_file = open(JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_PATH)
    compile(loaded_model)
    return loaded_model
