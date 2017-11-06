"""
Tools for reading data and feeding it into a model.
"""

# pylint: disable=E0611,E0401,E1101

import glob
from hashlib import md5
import json
import os
import random

from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.contrib.data import Dataset

from .images import IMAGE_SIZE

def hotspot_dataset(data_dir, num_timestamps=5):
    """
    Create a dataset for training a HotspotPredictor.

    Args:
      data_dir: a DataDir.
      num_timestamps: the number of timestamps to per video.

    Returns:
      A shuffled TensorFlow Dataset of (images, intensities), where images
        and intensities are batches.
    """
    generator_fn = lambda: data_dir.hotspot_data(num_timestamps=num_timestamps)
    dataset = Dataset.from_generator(generator_fn,
                                     (tf.string, tf.float32),
                                     output_shapes=((num_timestamps,), (num_timestamps,)))
    def read_images_fn(image_paths, labels):
        """
        Read the images from the paths.
        """
        tensors = [_read_image(image_paths[i]) for i in range(num_timestamps)]
        return tf.stack(tensors), labels
    return dataset.map(read_images_fn)

def popularity_dataset(data_dir):
    """
    Create a dataset for training a PopularityPredictor.

    Args:
      data_dir: A DataDir.

    Returns:
      A shuffled TensorFlow Dataset of (image, like_frac, views) tuples.
    """
    def generator_fn():
        """
        Generate (image_path, like_frac, views) tuples.
        """
        for thumbnail, _, metadata in data_dir.all_thumbnails():
            like_frac = metadata['votes_up'] / (metadata['votes_up'] + metadata['votes_down'])
            yield thumbnail, like_frac, metadata['views']
    dataset = Dataset.from_generator(generator_fn, (tf.string, tf.float32, tf.float32))
    return dataset.shuffle(buffer_size=20000).map(_read_image_in_tuple)

def category_dataset(data_dir, labels):
    """
    Create a dataset for training a CategoryTagger.

    Args:
      data_dir: A DataDir.
      labels: the string class labels to use.

    Returns:
      A shuffled TensorFlow Dataset of (image, categories) pairs.
    """
    def generator_fn():
        """
        Generate (image_path, category_vector) pairs.
        """
        for thumbnail, _, metadata in data_dir.all_thumbnails():
            bitmask = [l in metadata['categories'] for l in labels]
            yield thumbnail, bitmask
    dataset = Dataset.from_generator(generator_fn, (tf.string, tf.bool),
                                     output_shapes=((), (len(labels),)))
    return dataset.shuffle(buffer_size=20000).map(_read_image_in_tuple)

class DataDir:
    """
    A handle on a data directory.

    The directory should contain one sub-directory per video.
    Each sub-directory contains a metadata.json file and a set of files by the
    name of thumbnail_x.png, where x is a timestamp in seconds.

    Works with the directory structure produced by:
    https://github.com/aiporn/pornhub-scraper.
    """
    def __init__(self, path, validation=False):
        """
        Read a data directory.

        Args:
          path: path to the directory of data.
          validation: if True, load the validation set. Otherwise, load the
            training set.
        """
        self._root_dir = path
        self._id_to_meta = {}
        self._id_to_path = {}
        for entry in os.listdir(path):
            dir_path = os.path.join(path, entry)
            json_path = os.path.join(dir_path, 'metadata.json')
            if (entry.startswith('.') or not os.path.isdir(dir_path) or
                    not os.path.exists(json_path)):
                continue
            with open(json_path, 'r') as json_file:
                metadata = json.load(json_file)
            if _is_validation(metadata['id']) == validation:
                self._id_to_meta[metadata['id']] = metadata
                self._id_to_path[metadata['id']] = dir_path
        self._sorted_ids = sorted(list(self._id_to_path.keys()))

    @property
    def video_ids(self):
        """
        Get all the video IDs in the dataset.
        """
        return self._sorted_ids

    def all_thumbnails(self):
        """
        Iterate over all the thumbnails in the dataset.

        Returns:
          An iterator over tuples with three entries:
            thumbnail_path: path to thumbnail image.
            thumbnail_time: time of thumbnail in second.
            metadata: video metadata.

        The metadata contains the following keys:
          'duration': video duration in seconds.
          'categories': a list of category strings.
          'tags': a list of tag strings.
          'views': the number of video views.
          'votes_up': the number of upvotes.
          'votes_down': the number of downvotes.
          'hotspots': (possibly missing) hotspot array.
        """
        for video_id in self.video_ids:
            metadata = self._id_to_meta[video_id]
            for thumb_info in self.video_thumbnails(video_id):
                yield thumb_info + (metadata,)

    def video_thumbnails(self, video_id):
        """
        Iterate over all the thumbnails of a video.

        Returns:
          An iterator of tuples with two entries:
            thumbnail_path: path to thumbnail image.
            thumbnail_time: timestamp of the thumbnail.
        """
        thumbs = glob.glob(os.path.join(self._id_to_path[video_id], 'thumbnail_*.png'))
        for thumb_path in thumbs:
            timestamp = int(thumb_path.split('_')[-1])
            yield thumb_path, timestamp

    def hotspot_data(self, num_timestamps=5):
        """
        Iterate over random batches for hotspot training.

        Returns:
          An iterator of tuples with two entries:
            thumbnail_paths: paths to num_timestamps thumbnails.
            intensities: a list of intensities, one per thumbnail.
        """
        while True:
            video_id = random.choice(self.video_ids)
            hotspot_func = self.hotspot_function(video_id)
            thumbnails = [th for th in self.video_thumbnails(video_id)]
            if hotspot_func is None or len(thumbnails) < num_timestamps:
                continue
            while len(thumbnails) > num_timestamps:
                del thumbnails[random.randrange(len(thumbnails))]
            paths, timestamps = zip(*thumbnails)
            yield list(paths), [hotspot_func(t) for t in timestamps]

    def hotspot_function(self, video_id):
        """
        Get a function for the hotspot curve of a video.

        Returns:
          None if the video has no hotspots, or a callable that takes a
          timestamp and returns a view count.
        """
        metadata = self._id_to_meta[video_id]
        if metadata['hotspots'] is None:
            return None
        times = [i * metadata['duration'] / len(metadata['hotspots'])
                 for i, _ in enumerate(metadata['hotspots'])]
        counts = metadata['hotspots']
        return interp1d(times, counts)

def _read_image_in_tuple(path, *args):
    """
    Read the image path and keep the other data unchanged.
    """
    return (_read_image(path),) + tuple(args)

def _read_image(path):
    """
    Create a TensorFlow image from the image path.

    Automatically scales the image and does data augmentation.
    """
    data = tf.read_file(path)
    raw = tf.image.decode_image(data, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(raw, IMAGE_SIZE, IMAGE_SIZE)
    float_image = tf.cast(image, tf.float32) / 0xff
    noise = tf.constant([0.0148366, 0.01253134, 0.01040762], dtype=tf.float32)
    return float_image + noise * tf.random_normal(())

def _is_validation(video_id):
    """
    Check if a video ID should be sent to the validation set.

    The validation set is defined as the videos such that the first digit of
    the MD5 hash of the ID is 0 or 1.
    """
    hasher = md5()
    hasher.update(bytes(video_id, 'utf-8'))
    first = hasher.hexdigest()[0]
    return first in ['0', '1']
