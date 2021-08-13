#!/usr/bin/env python3
"""This module contains a function that builds, trains, and saves a neural
network classifier"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """This function builds, trains, and saves a neural network classifier"""

    xtra = X_train
    tray = Y_train
    valx = X_valid
    valy = Y_valid
    lays = layer_sizes
    reps = iterations
    rpt = 100

    x, y = create_placeholders(xtra.shape[1], tray.shape[1])
    pred = forward_prop(x, lays, activations)
    loss = calculate_loss(y, pred)
    acc = calculate_accuracy(y, pred)
    top = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', acc)
    tf.add_to_collection('train_op', top)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(reps + 1):
            tacc, losst = sess.run([acc, loss], {x: xtra, y: tray})
            vacc, vloss = sess.run([acc, loss], {x: valx, y: valy})
            if i % rpt == 0 or i == reps:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(losst))
                print("\tTraining Accuracy: {}".format(tacc))
                print("\tValidation Cost: {}".format(vloss))
                print("\tValidation Accuracy: {}".format(vacc))
            if i < reps:
                sess.run(top, {x: xtra, y: tray})
        save_path = saver.save(sess, save_path)
    return save_path
