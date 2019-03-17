#%%
""" https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d 
	"""
import keras
import matplotlib.pyplot as plt

import ElapsedTime as Timer


#Load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if __name__ == "__main__":
    image_index = 54050 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()