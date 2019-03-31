#%%

from keras.layers import Dropout, Dense, Flatten, Reshape, Input, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adadelta,Adam,SGD, RMSprop
from keras.activations import relu

import numpy as np

class discriminator_model(object):
    """
    builds a discriminator model, that tells how real an image is
    """
    def __init__(self, optimizer='Adam', inputshape=(28,28,1)):
        self.optimizer_choice = optimizer
        self.inputshape = inputshape

        if(self.optimizer_choice == 'Adam'):
            self.optimizer = Adam()

        if(self.optimizer_choice == 'RMSprop'):
            self.optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        ### insert other optimizer choices here

        conf = dict()
        conf["activation"] = "relu"
        conf["padding"] = "same"

        print('This model has the following Shapes:')
        inputLayer = Input(shape=self.inputshape)
        print('Input-Shape: ',inputLayer.shape)

        #statt batchnorm wird fürs downsampling striding genutzt!
        x = Conv2D(64, 5, strides=2, **conf)(inputLayer)
        x = Dropout(0.4)(x)

        x = Conv2D(128, 5, strides=2, **conf)(x)
        x = Dropout(0.4)(x)

        x = Conv2D(256, 5, strides=2, **conf)(x)
        x = Dropout(0.4)(x)

        x = Conv2D(512, 5, **conf)(x)
        x = Dropout(0.4)(x) 

        x = Flatten()(x)

        #sigmoid, da Werte von 0.0 bis 1.0
        prediction = Dense(1, activation='sigmoid')(x)
        print('Output-Shape: ',prediction.shape)

        self.model = Model(inputs=inputLayer, outputs=prediction)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        #self.model.summary()

if __name__ == "__main__":
    model = discriminator_model()
    model.model.summary()

