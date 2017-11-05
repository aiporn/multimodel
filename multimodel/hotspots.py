"""
Video hotspot prediction.

This model takes several thumbnails from random parts of a video and predicts
their relative popularities.
"""

import tensorflow as tf

class HotspotPredictor:
    """
    A model for predicting hotspots in a video.

    A HotspotPredictor deals with a fixed number of timestamps per video.
    If that is N, then it expects the number of input images to be divisible
    by N.
    """
    def __init__(self, image_network, num_timestamps=5):
        batch_size = tf.shape(image_network)[0]
        feature_size = tf.shape(image_network)[1]
        sub_batch_shape = (batch_size//num_timestamps, num_timestamps, feature_size)
        split_batches = tf.reshape(image_network, sub_batch_shape)

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

    def loss(self, actual_intensities, rescale_fn=tf.log):
        """
        Compute the prediction loss.

        Args:
          actual_intensities: a batch of intensity histograms.
          rescale_fn: callable for getting the actual intensities into a
            neural-friendly range.

        Returns:
          A 0-D Tensor representing the mean cosine distance between the
            actual and target histograms.
        """
        actual = _subtract_mean(rescale_fn(actual_intensities))
        prods = _normalize_mag(self.predictions) * _normalize_mag(actual)
        return 1 + tf.negative(tf.reduce_mean(tf.reduce_sum(prods, axis=1)))

def _subtract_mean(vector_batch):
    """
    For each vector in a batch, subtract the mean.
    """
    means = tf.expand_dims(tf.reduce_mean(vector_batch, axis=1), axis=1)
    return vector_batch - means

def _normalize_mag(vector_batch):
    """
    For each vector in a batch, divide by the norm.
    """
    return vector_batch / tf.norm(vector_batch, axis=1, keep_dims=True)
