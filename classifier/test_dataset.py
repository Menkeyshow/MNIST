#%%
""" 
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d 
"""

import os

print('Make sure to use right WD:',os.getcwd())  # Prints the current working directory

import keras
import matplotlib.pyplot as plt
import classifier.util.ElapsedTime as ElapsedTime
import classifier.models.classifier_model as classifier_model

#Load Dataset 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#3dim ==> 4dim
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

if __name__ == "__main__":
    Timer = ElapsedTime.ElapsedTime()
    print(x_train[0].shape)
    model = classifier_model.model_object('Adam', inputshape=(28,28,1))
    model.model.fit(x=x_train, y=y_train, epochs=10)
    image_index = 54050 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()
    Timer.printTime()