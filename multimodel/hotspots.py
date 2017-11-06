"""
Video hotspot prediction.

This model takes several thumbnails from random parts of a video and predicts
their relative popularities.
"""

import numpy as np
import tensorflow as tf

class HotspotPredictor:
    """
    A model for predicting hotspots in a video.

    A HotspotPredictor deals with a fixed number of timestamps per video.
    If that is N, then it expects the number of input images to be divisible
    by N.
    """
    def __init__(self, image_network, num_timestamps=5):
        batch_size = tf.shape(image_network.features)[0]
        feature_size = image_network.features.get_shape()[1]
        sub_batch_shape = (batch_size//num_timestamps, num_timestamps, feature_size)
        split_batches = tf.reshape(image_network.features, sub_batch_shape)

        # For now, the intensity prediction at each
        # timestamp will be a simple dot product.
        # In other words, the timestamps are independent
        # except for mean subtraction.
        dots = tf.layers.dense(split_batches, 1, use_bias=False)
        predictions_2d = tf.reshape(dots, sub_batch_shape[:-1])
        self._predictions = _subtract_mean(predictions_2d)

    @property
    def predictions(self):
        """
        Get a 2-D Tensor which is a batch of intensity histograms.

        Every histogram is zero-centered.
        """
        return self._predictions

    def loss(self, actual_intensities,
             rescale_fn=lambda x: tf.log(tf.clip_by_value(x, 1, np.inf))):
        """
        Compute the prediction loss.

        Args:
          actual_intensities: a batch of intensity histograms.
          rescale_fn: callable for getting the actual intensities into a
            neural-friendly range.

        Returns:
          A 0-D Tensor representing the distance between the actual and target
            histograms.
        """
        actual = _subtract_mean(rescale_fn(actual_intensities))
        return tf.reduce_mean(tf.square(self.predictions - actual))

def _subtract_mean(vector_batch):
    """
    For each vector in a batch, subtract the mean.
    """
    means = tf.expand_dims(tf.reduce_mean(vector_batch, axis=1), axis=1)
    return vector_batch - means
