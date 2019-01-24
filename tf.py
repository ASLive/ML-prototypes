# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import model_from_json

# Helper libraries
import numpy as np
from graph import *

print("Tensorflow version "+tf.__version__)

# import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocess the data: scale to 0-1 values
train_images = train_images / 255.0
test_images = test_images / 255.0

# # display the first 25 images
# display(train_images, class_names, train_labels)

def make_model():
    # setup the layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    model.fit(train_images, train_labels, epochs=5)

    return save_model(model)

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    return model

def read_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # compile the model
    loaded_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return loaded_model

#model = make_model()
model = read_model()

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make preductions
predictions = model.predict(test_images)
# print "predicted:"
# print np.argmax(predictions[0])
# print "actual:"
# print test_labels[0]
