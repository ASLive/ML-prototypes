import numpy as np
from graph import *
from ml import *
from data import *

print("Tensorflow version "+tf.__version__)

if __name__ == "__main__":

    train_images, train_labels, test_images, test_labels, class_names = load_data()
    model = ml_model(train_images, train_labels)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:'+str(test_acc))

    predictions = model.predict(test_images)

    display_single_prediction(predictions, test_labels, test_images, class_names, 0)
    display_single_prediction2(test_labels,class_names,model,test_images[0])
    display_multiple_prediction(predictions, test_labels, test_images, class_names)
