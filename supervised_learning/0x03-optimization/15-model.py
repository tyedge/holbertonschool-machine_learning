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
    stat = np.random.get_state()
    Xuffle = np.random.permutation(X)
    np.random.set_state(stat)
    Yuffle = np.random.permutation(Y)
    return Xuffle, Yuffle


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """This function creates a learning rate decay operation in tensorflow
using inverse time decay"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def create_batch_norm_layer(prev, n, activation):
    """This function creates a batch normalization layer for a neural network
in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, kernel_initializer=init, activation=None)
    x = lay(prev)
    mean, var = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    out = tf.nn.batch_normalization(x, mean=mean, variance=var,
                                    offset=beta, scale=gamma,
                                    variance_epsilon=1e-8)
    if activation is None:
        return out
    return activation(out)


def create_placeholders(nx, classes):
    """This function returns placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """This function returns tensor output of the layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """This function creates the forward propagation graph
for the neural network"""
    pred = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            pred = create_batch_norm_layer(pred, layer_sizes[i],
                                           activations[i])
        else:
            pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred


def calculate_accuracy(y, y_pred):
    """This function calculates the accuracy of a prediction"""
    labs = tf.argmax(y, axis=1)
    preds = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labs), dtype=tf.float32))


def calculate_loss(y, y_pred):
    """This function calculates the softmax cross-entropy loss of a
prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """This function builds, trains, and saves a neural network model in
tensorflow using Adam optimization, mini-batch gradient descent, learning rate
decay, and batch normalization"""
    x = tf.placeholder(tf.float32, shape=(None, Data_train[0].shape[1]),
                       name="x")
    y = tf.placeholder(tf.float32, shape=(None, Data_train[1].shape[1]),
                       name="y")
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    accur = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accur)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    _step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, _step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        train_dict = {x: Data_train[0], y: Data_train[1]}
        val_dict = {x: Data_valid[0], y: Data_valid[1]}
        reps = Data_train[0].shape[0] / batch_size
        iters = int(reps)
        if reps > iters:
            iters = int(reps) + 1
            extra = True
        else:
            extra = False
        for i in range(epochs + 1):
            acct, losst = sess.run([accur, loss], train_dict)
            vacc, vloss = sess.run([accur, loss], val_dict)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(losst))
            print("\tTraining Accuracy: {}".format(acct))
            print("\tValidation Cost: {}".format(vloss))
            print("\tValidation Accuracy: {}".format(vacc))
            if i < epochs:
                Xuffle, Yuffle = shuffle_data(Data_train[0], Data_train[1])
                for j in range(iters):
                    start = j * batch_size
                    if j == iters - 1 and extra:
                        end = (int(start + (reps - iters + 1) * batch_size))
                    else:
                        end = start + batch_size
                    mini_dict = {x: Xuffle[start: end], y: Yuffle[start: end]}
                    sess.run(train_op, mini_dict)
                    if (j + 1) % 100 == 0 and j != 0:
                        print("\tStep {}:".format(j + 1))
                        miniacc, minico = sess.run([accur, loss], mini_dict)
                        print("\t\tCost: {}".format(minico))
                        print("\t\tAccuracy: {}".format(miniacc))
                sess.run(tf.assign(_step, _step + 1))
        save_path = saver.save(sess, save_path)
    return save_path
