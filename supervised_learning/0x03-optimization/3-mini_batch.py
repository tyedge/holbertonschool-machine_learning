#!/usr/bin/env python3
"""This module contains a function that trains a loaded neural network model
using mini-batch gradient descent"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """This function trains a loaded neural network model using mini-batch
gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accur = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
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

                for k in range(1, iters + 1):

                    sess.run(train_op, {x: Xert[start:stop],
                                        y: Yert[start:stop]})

                    if not k % 100 and x != 0:
                        rep = X_train.shape[0] % batch_size
                        accure, losse = sess.run([accur, loss],
                                                 {x: Xert[start:stop],
                                                  y: Yert[start:stop]})

                        print("\tStep {}:".format(k))
                        print("\t\tCost: {}".format(losse))
                        print("\t\tAccuracy: {}".format(accure))

                        start += batch_size
                        if k + 1 == iters and rep != 0:
                            stop = start + rep
                        else:
                            stop = start + batch_size

        save_path = saver.save(sess, save_path)

    return save_path
