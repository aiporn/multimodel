"""
Video popularity prediction.

This model takes a video thumbnail and predicts the views and likes.
"""

import numpy as np
import tensorflow as tf

class PopularityPredictor:
    """
    A model for predicting video popularity.
    """
    def __init__(self, image_network):
        predictions = tf.layers.dense(image_network.features, 2)
        self._like_frac = tf.nn.sigmoid(predictions[:, 0])
        self._views = predictions[:, 1]

    @property
    def like_frac(self):
        """
        Get a 1-D Tensor of like fraction predictions.

        Like fractions are numbers between 0 and 1.
        """
        return self._like_frac

    @property
    def views(self):
        """
        Get a 1-D Tensor of view count predictions.
        """
        return self._views

    def loss(self, actual_like_frac, actual_views,
             rescale_fn=lambda x: tf.clip_by_value(tf.log(x), 1, np.inf)):
        """
        Compute the prediction loss.

        Args:
          actual_like_frac: ground-truth like fractions.
          actual_views: ground-truth view counts.
          rescale_fn: callable for getting the actual view counts into a
            neural-friendly range.

        Returns:
          A 0-D Tensor representing the mean prediction loss.
        """
        like_loss = tf.reduce_mean(tf.square(actual_like_frac - self.like_frac))
        view_loss = tf.reduce_mean(tf.square(rescale_fn(actual_views) - self.views))
        return like_loss + view_loss
