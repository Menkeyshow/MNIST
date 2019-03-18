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
(x_train_pre, y_train), (x_test_pre, y_test) = keras.datasets.mnist.load_data()
#3dim ==> 4dim
x_train = x_train_pre.reshape(x_train_pre.shape[0], x_train_pre.shape[1], x_train_pre.shape[2], 1)
x_test = x_test_pre.reshape(x_test_pre.shape[0], x_test_pre.shape[1], x_test_pre.shape[2], 1)


if __name__ == "__main__":
    Timer = ElapsedTime.ElapsedTime()
    model = classifier_model.model_object('Adam', inputshape=(28,28,1))
    model.model.fit(x=x_train, y=y_train, epochs=10)
    model.model.evaluate(x=x_test, y=y_test)
    print('Trainingsduration:')
    Timer.printTime()

    #Test Result :D
    print('Die Zahl ist: ', y_train[54050])
    plt.imshow(x_train_pre[54050], cmap='Greys')
    zahl = model.model.predict(y_train[54050].reshape(1, 28, 28, 1))
    plt.show()
    print('Netz sagt: ', zahl)

    