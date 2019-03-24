#%%
""" 
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d 
"""

import os, sys

#for correct imports from "main" directory
module_path = os.path.abspath(os.getcwd())    
if module_path not in sys.path:       
    sys.path.append(module_path)

#for Unix/Windows compatible paths
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from classifier.util.ElapsedTime import ElapsedTime
import classifier.models.classifier_model as classifier_model

class classifier(object):
    def __init__(self, model, do_train=True, do_test=True, do_use_weights=False, do_save_weights=True, epochs=10):
        self.timer = ElapsedTime()
        
        self.do_train = do_train
        self.do_test = do_test
        self.do_use_weights = do_use_weights
        self.do_save_weights = do_save_weights
        self.epochs = epochs

        self.model = model
        self.isDataPrepared = False
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.weight_path = Path('weights/','%sepochs.h5' % self.epochs)

        if(self.do_use_weights):
            try:
                print('Looking for weight in: ',self.weight_path)
                self.model.model.load_weights(self.weight_path)
            except:
                print('Cant load saved weights, not yet trained?')
                sys.exit(1)
            print('Loaded saved weights')
    
    def prepare_data(self, x_train, y_train, x_test, y_test, isNormalized=False):
        #3dim ==> 4dim
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        print('Data now reshaped into:', x_train.shape,'and', x_test.shape)

        #normalize values between 0 and 1
        if(not isNormalized):
            x_train = x_train.astype('float32')
            x_train /= 255

            x_test = x_test.astype('float32')
            x_test /= 255
            print('Data now normalized!')
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.isDataPrepared = True
        
    def train_model(self):
        trainTimer = ElapsedTime()
        print('Begin training!')

        if(not self.isDataPrepared):
            print('Data wasnt prepared before training/evaluating, aborting...')
            return
        
        self.model.model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs)

        if (self.do_save_weights):
            self.model.model.save_weights(self.weight_path)
    
        print('Done training!')
        trainTimer.printTime()

    def evaluate_model(self):
        evalTimer = ElapsedTime()
        print('Begin evaluating!')

        scores = self.model.model.evaluate(x=self.x_test, y=y_test)

        print('Done evaluating!')
        evalTimer.printTime()
        return scores


if __name__ == "__main__":
    # Load dataset MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Set up objects
    model = classifier_model.model_object('Adam', inputshape=(28,28,1))
    classifier = classifier(model, epochs=5)

    # Train classifier
    classifier.prepare_data(x_train, y_train, x_test, y_test, isNormalized=False)
    classifier.train_model()

    # Evaluate classifier
    scores = classifier.evaluate_model()
    print('Script-Duration: ', classifier.timer.getTime())
    print(scores, '\n', classifier.model.model.metrics_names)

    # Test Result :D
    print('Die Zahl ist: ', y_train[54050])
    plt.imshow(x_train[54050], cmap='Greys')
    prediction = classifier.model.model.predict(x_train[54050].reshape(1, 28, 28, 1))
    plt.show()
    print('Inhalt des Arrays: ',prediction)
    print('Netz sagt: ', np.argmax(prediction))

    