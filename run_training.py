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

    print('Sizes: train=%d validation=%d' %
          (len(training.video_ids), len(validation.video_ids)))

    print('Creating models...')
    train_feed = {}
    with tf.variable_scope('aggregate'):
        agg, train_loss = aggregate_from_data(training, loss_scope='train_loss')
        train_feed.update(agg.training_feed_dict())
    with tf.variable_scope('aggregate', reuse=True):
        agg, validation_loss = aggregate_from_data(validation, loss_scope='test_loss')
        train_feed.update(agg.training_feed_dict())
    cur_iter = tf.assign_add(tf.Variable(0, dtype=tf.int32, name='cur_iter'),
                             tf.constant(1, dtype=tf.int32))
    minim = make_minimizer(train_loss, cur_iter)

    print('Training...')
    training_loop(train_loss, validation_loss, cur_iter, minim, train_feed)

def make_minimizer(loss, cur_iter):
    """
    Create a minimizer that optimizes the loss.

    The learning rate may decay based on the iteration number.
    """
    rate = 1e-3 * tf.pow(0.9999, tf.cast(cur_iter, tf.float32))
    optim = tf.train.AdamOptimizer(learning_rate=rate)
    return optim.minimize(loss)

def training_loop(train_loss, validation_loss, cur_iter, minimize, feed_dict):
    """
    Run the inner training loop.
    """
    saver = tf.train.Saver()
    all_summaries = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_from_checkpoint(sess, saver)
        summary_writer = tf.summary.FileWriter(SAVE_DIR, sess.graph)
        while True:
            terms = (train_loss, validation_loss, cur_iter, minimize, all_summaries)
            train, validation, iter_num, _, summary = sess.run(terms, feed_dict=feed_dict)
            print('iter %d: train_loss=%f val_loss=%f' % (iter_num, train, validation))
            summary_writer.add_summary(summary, iter_num)
            if iter_num % SAVE_INTERVAL == 0:
                saver.save(sess, os.path.join(SAVE_DIR, 'model'), global_step=iter_num)

def restore_from_checkpoint(sess, saver):
    """
    Restore the model from the latest checkpoint if there
    was one.
    """
    latest = tf.train.latest_checkpoint(SAVE_DIR)
    if latest:
        saver.restore(sess, latest)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python run_training.py <data dir>\n')
        sys.exit(1)
    main(sys.argv[1])
