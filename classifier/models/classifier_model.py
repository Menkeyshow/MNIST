#%%

from keras.layers import Flatten, Reshape, Input, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adadelta,Adam,SGD

import numpy as np

class model_object(object):
    """
    builds a model object
    """
    def __init__(self, optimizer='Adam', inputshape=(26,26,1)):
        self.optimizer_choice = optimizer
        self.inputshape = inputshape

        if(self.optimizer_choice == 'Adam'):
            self.optimizer=Adam()
        
        conf = dict()
        conf["activation"] = "relu"
        conf["padding"] = "same"
    
        inputLayer = Input(shape=self.inputshape)
        print(inputLayer.shape)
        x = Conv2D(16, 3, **conf)(inputLayer)
        x = BatchNormalization()(x)

        x = Conv2D(32, 3, **conf)(x)
        x = BatchNormalization()(x)

        x = Conv2D(32, 3, **conf)(x)
        x = BatchNormalization()(x)

        x = Conv2D(16, 3, **conf)(x)
        x = BatchNormalization()(x)

        x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)
        predictions = Reshape((-1,))(x)


        self.model = Model(inputs=inputLayer, outputs=predictions)
        self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    model = model_object()