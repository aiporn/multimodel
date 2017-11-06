"""
An all-in-one model.
"""

# pylint: disable=E1129

import tensorflow as tf

from .data import hotspot_dataset, popularity_dataset, category_dataset
from .hotspots import HotspotPredictor
from .images import IMAGE_SIZE, ImageNetwork
from .popularity import PopularityPredictor
from .tagging import CategoryTagger, all_categories

def aggregate_from_data(data, num_timestamps=5, batch_size=32):
    """
    Create an Aggregate for training on the given DataDir.

    Args:
      data: a DataDir for training.
      num_timestamps: number of timestamps for hotspot model.
      batch_size: max number of images per batch.

    Returns:
      A tuple containing the following:
        aggregate: the aggregate model.
        loss: a Tensor representing the loss of the model for the batch.
    """
    hotspots = hotspot_dataset(data, num_timestamps=num_timestamps)
    hotspots = hotspots.repeat()
    hotspots = hotspots.batch(batch_size//num_timestamps)
    hotspot_in, intensities = hotspots.make_one_shot_iterator().get_next()

    popularity = popularity_dataset(data).repeat().batch(batch_size)
    popularity_in, like_fracs, views = popularity.make_one_shot_iterator().get_next()

    categories = category_dataset(data, all_categories()).repeat().batch(batch_size)
    categories_in, categories_out = categories.make_one_shot_iterator().get_next()

    agg = Aggregate(hotspot_images=_flatten_image_batches(hotspot_in),
                    popularity_images=popularity_in,
                    category_images=categories_in,
                    num_timestamps=num_timestamps)
    loss = agg.loss(intensities, categories_out, like_fracs, views)
    return agg, loss

class Aggregate:
    """
    A combined machine learning model for porn analysis.
    """
    def __init__(self, hotspot_images=None, popularity_images=None, category_images=None,
                 num_timestamps=5):
        """
        Create an aggregate model.

        Args:
          hotspot_images: batch of images for hotspot analysis.
            If None, a new placeholder is created.
          popularity_images: batch of images for popularity prediction.
            If None, a new placeholder is created.
          category_images: batch of images for category tagging.
            If None, a new placeholder is created.
          num_timestamps: number of timesteps for hotspots.
        """
        self._hotspot_images = _image_or_placeholder(hotspot_images)
        with tf.variable_scope('image_network'):
            image_network = ImageNetwork(self._hotspot_images)
        with tf.variable_scope('hotspots'):
            self._hotspot_model = HotspotPredictor(image_network, num_timestamps=num_timestamps)

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

def _flatten_image_batches(imgs):
    """
    Turn a batch of image batches into one image batch.
    """
    old_shape = tf.shape(imgs)
    new_shape = tf.concat([tf.expand_dims(old_shape[0]*old_shape[1], 0), old_shape[2:]],
                          axis=0)
    return tf.reshape(imgs, new_shape)
