"""
An all-in-one model.
"""

# pylint: disable=E1129

import tensorflow as tf

from .hotspots import HotspotPredictor
from .images import IMAGE_SIZE, ImageNetwork
from .popularity import PopularityPredictor
from .tagging import CategoryTagger

class Aggregate:
    """
    A combined machine learning model for porn analysis.
    """
    def __init__(self, hotspot_images=None, popularity_images=None, category_images=None):
        """
        Create an aggregate model.

        Args:
          hotspot_images: batch of images for hotspot analysis.
            If None, a new placeholder is created.
          popularity_images: batch of images for popularity prediction.
            If None, a new placeholder is created.
          category_images: batch of images for category tagging.
            If None, a new placeholder is created.
        """
        self._hotspot_images = _image_or_placeholder(hotspot_images)
        with tf.variable_scope('image_network'):
            image_network = ImageNetwork(self._hotspot_images)
        with tf.variable_scope('hotspots'):
            self._hotspot_model = HotspotPredictor(image_network)

        self._popularity_images = _image_or_placeholder(popularity_images)
        with tf.variable_scope('image_network', reuse=True):
            image_network = ImageNetwork(self._popularity_images)
        with tf.variable_scope('popularity'):
            self._popularity_model = PopularityPredictor(image_network)

        self._category_images = _image_or_placeholder(category_images)
        with tf.variable_scope('image_nework', reuse=True):
            image_network = ImageNetwork(self._category_images)
        with tf.variable_scope('categories'):
            self._category_model = CategoryTagger(image_network)

    @property
    def hotspot_images(self):
        """
        Get the input Tensor for the hotspot model.
        """
        return self._hotspot_images

    @property
    def popularity_images(self):
        """
        Get the input Tensor for the popularity model.
        """
        return self._popularity_images

    @property
    def category_images(self):
        """
        Get the input Tensor for the category tagging model.
        """
        return self._category_images

    @property
    def hotspot_model(self):
        """
        Get the hotspot model.
        """
        return self._hotspot_model

    @property
    def popularity_model(self):
        """
        Get the popularity model.
        """
        return self._popularity_model

    @property
    def category_model(self):
        """
        Get the category tagging model.
        """
        return self._category_model

    def loss(self, hotspot_intensities, categories, like_fracs, views):
        """
        Compute a joint loss for all the models.

        Args:
          hotspot_intensities: ground-truth hotspot values.
          categories: ground-truth category labels.
          like_fracs: ground-truth like fractions.
          views: ground-truth view counts.

        Returns:
          A 0-D Tensor representing the joint loss.
        """
        return (self.hotspot_model.loss(hotspot_intensities) +
                self.category_model.loss(categories) +
                self.popularity_model.loss(like_fracs, views))

def _image_or_placeholder(imgs):
    """
    Create a placeholder for the images if they are None.
    """
    if imgs is None:
        return tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    return imgs
