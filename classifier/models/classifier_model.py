#%%

from keras.layers import Dropout, Dense, Flatten, Reshape, Input, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adadelta,Adam,SGD

import numpy as np

class model_object(object):
    """
    builds a model object
    """
    def __init__(self, optimizer='Adam', inputshape=(28,28,1)):
        self.optimizer_choice = optimizer
        self.inputshape = inputshape

        if(self.optimizer_choice == 'Adadelta'):
            self.optimizer=Adadelta()

        if(self.optimizer_choice == 'Adam'):
            self.optimizer=Adam()

        if(self.optimizer_choice == 'SGD'):
            self.optimizer=SGD()
        
        conf = dict()
        conf["activation"] = "relu"
        conf["padding"] = "same"
    
        print('This model has the following Shapes:')
        inputLayer = Input(shape=self.inputshape)
        print('Input-Shape: ',inputLayer.shape)
        
        x = Conv2D(28, 3, **conf)(inputLayer)
        #x = BatchNormalization()(x)

        x = MaxPooling2D(padding="same")(x)

        x = Flatten()(x)

        x = Dense(128, activation='relu')(x)

        x = Dropout(rate=0.2)(x)

        predictions = Dense(10, activation="softmax")(x)
        
        print('Output-Shape: ',predictions.shape)


        self.model = Model(inputs=inputLayer, outputs=predictions)
        #welche Loss Fucntion????? wtf nur sparse_categorical funktioniert...
        #https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        self.model.compile(optimizer=self.optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # def fit(self, train, label, epochs=10):
    #     self.model.fit(train, label, epochs=epochs)

    # def evaluate(self, test, label):
    #     self.model.evaluate(test, label)
if __name__ == "__main__":
    model = model_object()