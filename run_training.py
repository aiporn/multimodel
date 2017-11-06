"""
Train a model on some data.

You will likely want a GPU to run this.
"""

import os
import sys

import tensorflow as tf

from multimodel import DataDir, aggregate_from_data

SAVE_DIR = 'trained_model'
SAVE_INTERVAL = 100

def main(data_path):
    """
    Kick off training.
    """
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    print('Loading training data...')
    training = DataDir(data_path)

    print('Loading validation data...')
    validation = DataDir(data_path, validation=True)

    print('Creating models...')
    with tf.variable_scope('aggregate'):
        _, train_loss = aggregate_from_data(training)
    with tf.variable_scope('aggregate', reuse=True):
        _, validation_loss = aggregate_from_data(validation)
    cur_iter = tf.assign_add(tf.Variable(0, dtype=tf.int32, name='cur_iter'), 1)
    minim = make_minimizer(train_loss, cur_iter)

    training_loop(train_loss, validation_loss, cur_iter, minim)

def make_minimizer(loss, cur_iter):
    """
    Create a minimizer that optimizes the loss.

    The learning rate may decay based on the iteration number.
    """
    rate = 1e-3 * tf.pow(0.9999, cur_iter)
    optim = tf.train.AdamOptimizer(learning_rate=rate)
    return optim.minimize(loss)

def training_loop(train_loss, validation_loss, cur_iter, minimize):
    """
    Run the inner training loop.
    """
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latest = tf.train.latest_checkpoint(SAVE_DIR)
        if latest:
            saver.restore(sess, latest)
        while True:
            terms = (train_loss, validation_loss, cur_iter, minimize)
            train, validation, iter_num, _ = sess.run(terms)
            print('iter %d: train_loss=%f val_loss=%f' % (iter_num, train, validation))
            if iter_num % SAVE_INTERVAL == 0:
                saver.save(sess, os.path.join(SAVE_DIR, 'model'), global_step=iter_num)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python run_training.py <data dir>\n')
        sys.exit(1)
    main(sys.argv[1])
