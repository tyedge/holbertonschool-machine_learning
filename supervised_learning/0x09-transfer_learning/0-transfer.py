#!/usr/bin/env python3
"""This module contains a function that pre-processes the data for a cnn
model and a python script that trains a convolutional neural network to
classify the CIFAR10 dataset"""

import tensorflow as tf
import tensorflow.keras as K


def preprocess_data(X, Y):
    """This function pre-processes data for a cnn model"""
    Xpre = K.applications.inception_resnet_v2.preprocess_input(X)
    Ypre = K.utils.to_categorical(Y, 10)
    return Xpre, Ypre


(Xtrain, Ytrain), (Xtest, Ytest) = K.datasets.cifar10.load_data()

Xtrain, Ytrain = preprocess_data(Xtrain, Ytrain)
Xtest, Ytest = preprocess_data(Xtest, Ytest)

base = K.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                        input_shape=(299, 299, 3))
inputs = K.Input(shape=(32, 32, 3))
scale = K.layers.Lambda(lambda imgs: tf.image.resize(imgs, (299, 299)))(inputs)
blay = base(scale, training=False)
lay = K.layers.GlobalAveragePooling2D()(blay)
lay = K.layers.Dense(500, activation="relu")(lay)
lay = K.layers.Dropout(0.3)(lay)
out = K.layers.Dense(10, activation="softmax")(lay)
model = K.Model(inputs, out)

base.trainable = False

model.compile(optimizer=K.optimizers.Adam(), metrics=["acc"],
              loss='categorical_crossentropy')

model.fit(Xtrain, Ytrain, batch_size=300, epochs=4,
          validation_data=(Xtest, Ytest), verbose=1, shuffle=True)

model.save('cifar10.h5')
