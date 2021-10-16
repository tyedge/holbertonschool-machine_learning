#!/usr/bin/env python3
"""This module contains a funtion that builds, trains, and saves a neural
network model in tensorflow using Adam optimization, mini-batch gradient
descent, learning rate decay, and batch normalization"""

import tensorflow as tf
import numpy as np


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """This function creates the training operation for a neural network in
tensorflow using the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def shuffle_data(X, Y):
    """This function shuffles the data points in two matrices the same way"""
    shuffler = np.random.permutation(X.shape[0])
    shufflex = X[shuffler]
    shuffley = Y[shuffler]
    return shufflex, shuffley


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """This function creates a learning rate decay operation in tensorflow
using inverse time decay"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def create_batch_norm_layer(prev, n, activation):
    """This function creates a batch normalization layer for a neural network
in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, kernel_initializer=init)
    mean, var = tf.nn.moments(lay(prev), axes=[0])
    gamma = tf.get_variable(name="gamma", shape=[n],
                            initializer=tf.ones_initializer(), trainable=True)
    beta = tf.get_variable(name="beta", shape=[n],
                           initializer=tf.zeros_initializer(), trainable=True)
    out = tf.nn.batch_normalization(lay(prev), mean=mean, variance=var,
                                    offset=beta, scale=gamma,
                                    variance_epsilon=1e-8)
    return activation(out)


def create_placeholders(nx, classes):
    """This function returns placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """This function returns tensor output of the layer"""
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    out = tf.layers.Dense(n, activation=activation, kernel_initializer=weight,
                          name="layer")
    return out(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """This function creates the forward propagation graph
for the neural network"""
    pred = x
    for i in range(len(layer_sizes)):
        pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred


def calculate_accuracy(y, y_pred):
    """This function calculates the accuracy of a prediction"""
    labs = tf.argmax(y, axis=1)
    preds = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(labs, preds), dtype=tf.float32))


def calculate_loss(y, y_pred):
    """This function calculates the softmax cross-entropy loss of a
prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def train_mini_batch(sess, X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, save_path="/tmp/model.ckpt"):
    """This function trains a loaded neural network model using mini-batch
gradient descent"""

    saver = tf.train.Saver()
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    accur = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    alpha = tf.get_collection('alpha')[0]
    _step = tf.get_collection('step')[0]

    for i in range(epochs + 1):
        sess.run(_step.assign(i))
        sess.run(alpha)
        acct, losst = sess.run([accur, loss], {x: X_train, y: Y_train})
        vacc, vloss = sess.run([accur, loss], {x: X_valid, y: Y_valid})
        print("After {} epochs:".format(i))
        print("\tTraining Cost: {}".format(losst))
        print("\tTraining Accuracy: {}".format(acct))
        print("\tValidation Cost: {}".format(vloss))
        print("\tValidation Accuracy: {}".format(vacc))

        if i < epochs:
            iters = (X_train.shape[0] // batch_size + 1)
            Xert, Yert = shuffle_data(X_train, Y_train)
            start = 0
            stop = start + batch_size

            for j in range(1, iters + 1):
                sess.run(train_op, {x: Xert[start:stop],
                                    y: Yert[start:stop]})

                if j % 100 == 0 and j != 0:
                    rep = X_train.shape[0] % batch_size
                    if rep != 0:
                        iters += 1
                    accure, losse = sess.run([accur, loss],
                                             {x: Xert[start:stop],
                                              y: Yert[start:stop]})
                    print("\tStep {}:".format(j))
                    print("\t\tCost: {}".format(losse))
                    print("\t\tAccuracy: {}".format(accure))

                start += batch_size
                if j + 1 == iters and rep != 0:
                    stop = start + rep
                else:
                    stop = start + batch_size
        save_path = saver.save(sess, save_path)
    return save_path


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """This function builds, trains, and saves a neural network model in
tensorflow using Adam optimization, mini-batch gradient descent, learning rate
decay, and batch normalization"""
    X_t, Y_t = Data_train
    X_v, Y_v = Data_valid
    shp = X_t.shape[1]
    cls = Y_t.shape[1]

    x, y = create_placeholders(shp, cls)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    accur = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accur)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    _step = tf.Variable(0, trainable=False)
    tf.add_to_collection('step', _step)

    alpha = learning_rate_decay(alpha, decay_rate, _step, 1)
    tf.add_to_collection('alpha', alpha)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        save_path = train_mini_batch(sess, X_t, Y_t, X_v, Y_v, batch_size,
                                     epochs, save_path)
    return save_path
