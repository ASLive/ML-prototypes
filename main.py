import numpy as np
from graph import *
from ml import *

print("Tensorflow version "+tf.__version__)


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
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [train_images, train_labels, test_images, test_labels, class_names]


if __name__ == "__main__":

    train_images, train_labels, test_images, test_labels, class_names = load_data()
    model = ml_model(train_images, train_labels)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:'+str(test_acc))

    predictions = model.predict(test_images)

    display_single_prediction(predictions, test_labels, test_images, class_names, 0)
    display_single_prediction2(test_labels,class_names,model,test_images[0])
    display_multiple_prediction(predictions, test_labels, test_images, class_names)
