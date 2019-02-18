import numpy as np
from graph import *
from ml import *
from data import *
import sys

print("Tensorflow version "+tf.__version__)

if __name__ == "__main__":
    """
       python3 main.py # load model files if available, else retrain and save
       python3 main.py retrain # retrain model and save
       python3 main.py retrain no-save # retrain the model but don't save model
    """
    retrain_arg = len(sys.argv) > 1 and (sys.argv[1].lower() == 'retrain')
    no_save_arg = not (len(sys.argv) > 2 and (sys.argv[2].lower() == 'no-save'))

    train_images, train_labels, test_images, test_labels, class_names = load_data()
    model = ml_model(train_images, train_labels, retrain_arg, no_save_arg)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    # train_loss, train_acc = model.evaluate(train_images, train_labels)

    print('Test accuracy:'+str(test_acc))
    # print('Train accuracy:'+str(train_acc))

    predictions = model.predict(test_images)

    display_single_prediction(predictions, test_labels, test_images, class_names, 0)
    display_single_prediction2(test_labels,class_names,model,test_images[0])
    display_multiple_prediction(predictions, test_labels, test_images, class_names)
