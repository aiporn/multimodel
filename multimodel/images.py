"""
Image processing models.
"""

# pylint: disable=E1129

import tensorflow as tf

IMAGE_SIZE = 224

class ImageNetwork:
    """
    The core image processing module, implemented as a convolutional neural
    network.
    """
    def __init__(self, image_batch):
        assert image_batch.get_shape()[-1] == 3
        assert image_batch.get_shape()[-2] == IMAGE_SIZE
        assert image_batch.get_shape()[-3] == IMAGE_SIZE
        self._training_ph = tf.placeholder_with_default(False, ())
        self._features = _resnet_34(image_batch, self._training_ph)

    @property
    def features(self):
        """
        Get the batch of feature vectors from the network.
        """
        return self._features

    def training_feed_dict(self):
        """
        Get a dict to feed into TF during training to indicate that the model
        is being trained.
        """
        return {self._training_ph: True}

def _resnet_34(inputs, training_ph):
    """
    Apply ResNet-34 to the inputs.

    See https://arxiv.org/abs/1512.03385.
    """
    outputs = _resnet_34_start_layers(inputs, training_ph)
    for size in [3, 3, 5]:
        outputs = _repeated_residual_block(outputs, size, training_ph)
        outputs = _downsampling_residual_block(outputs, training_ph)
    outputs = _repeated_residual_block(outputs, 2, training_ph)
    outputs = tf.reduce_mean(outputs, axis=-2)
    outputs = tf.reduce_mean(outputs, axis=-2)
    assert len(outputs.get_shape()) == 2
    return outputs

def _resnet_34_start_layers(inputs, training_ph):
    """
    Rapidly reduce the dimensionality of the inputs.
    """
    conv_out = tf.layers.conv2d(inputs, 64, 7, strides=2, padding='same')
    batch_norm = tf.layers.batch_normalization(conv_out, training=training_ph)
    pooled = tf.layers.max_pooling2d(batch_norm, 3, 2, padding='same')
    return tf.nn.relu(pooled)

def _repeated_residual_block(inputs, num_repeats, training_ph):
    """
    Apply num_repeats residual blocks.
    """
    num_filters = int(inputs.get_shape()[-1])
    with tf.variable_scope('residuals_' + str(num_filters)):
        for i in range(num_repeats):
            with tf.variable_scope('layer_' + str(i * 2)):
                conv_out = tf.layers.conv2d(inputs, num_filters, 3, padding='same')
                batch_norm = tf.layers.batch_normalization(conv_out, training=training_ph)
                out = tf.nn.relu(batch_norm)
            with tf.variable_scope('layer_' + str(i*2 + 1)):
                conv_out = tf.layers.conv2d(out, num_filters, 3, padding='same')
                batch_norm = tf.layers.batch_normalization(conv_out, training=training_ph)
            inputs = tf.nn.relu(inputs + batch_norm)
        return inputs

def _downsampling_residual_block(inputs, training_ph):
    """
    Double the depth but halve the width and height.
    """
    old_depth = int(inputs.get_shape()[-1])
    new_depth = old_depth * 2
    with tf.variable_scope('downsampling_' + str(old_depth)):
        with tf.variable_scope('proj'):
            proj_old = tf.layers.conv2d(inputs, new_depth, 1, strides=2, padding='same')
        with tf.variable_scope('layer_0'):
            conv_out = tf.layers.conv2d(inputs, new_depth, 3, strides=2, padding='same')
            batch_norm = tf.layers.batch_normalization(conv_out, training=training_ph)
            out = tf.nn.relu(batch_norm)
        with tf.variable_scope('layer_1'):
            conv_out = tf.layers.conv2d(out, new_depth, 3, padding='same')
            batch_norm = tf.layers.batch_normalization(conv_out, training=training_ph)
        return tf.nn.relu(proj_old + batch_norm)
