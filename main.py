import numpy as np
from graph import *
from ml import *

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

model = ml_model(train_images, train_labels)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make preductions
predictions = model.predict(test_images)
# print "predicted:"
# print np.argmax(predictions[0])
# print "actual:"
# print test_labels[0]
