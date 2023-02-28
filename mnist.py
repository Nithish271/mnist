import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from matplotlib import cm
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

(X_train, Y_train) , (X_test, Y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)    #training set
X_test = X_test.reshape(-1,28,28,1)      #test set

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)

input_shape=(28, 28, 1)
    # Three steps to create a CNN
    # 1. Convolution
    # 2. Activation
    # 3. Pooling
    # Repeat Steps 1,2,3 for adding more hidden layers

    # 4. After that make a fully connected network
    # This fully connected network gives ability to the CNN
    # to classify the samples

model = tf.keras.Sequential(
        (
            # 32 filters
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(axis=-1),
            # Pooling layer
             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # increase the number of filters
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            # 1st hidden layer in fully connected layer
            # 512 nodes
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        )
    )

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#model.fit(X_train, Y_train, shuffle=True, epochs=50, validation_split=0.2, verbose=2)

#model.save('mnist.h5')

model = tf.keras.models.load_model("mnist.h5")

img = X_test[0]
model.predict(img)