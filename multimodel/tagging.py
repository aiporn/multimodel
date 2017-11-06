"""
Single-image tagging models.
"""

from abc import ABC, abstractproperty

import tensorflow as tf

class ImageTagger(ABC):
    """
    A generic classifier that predicts a set of classes for an image.
    """
    def __init__(self, image_network):
        self._logits = tf.layers.dense(image_network.features, len(self.class_labels))
        self._probabilities = tf.nn.sigmoid(self._logits)
        self._classifications = tf.greater(self._logits, 0)

    @property
    def probabilities(self):
        """
        Get the predicted class probabilities.
        """
        return self._probabilities

    @property
    def classifications(self):
        """
        Get the predicted classifications (as tf.bools).
        """
        return self._classifications

    @abstractproperty
    def class_labels(self):
        """
        A string representation for each class.
        """
        pass

    def loss(self, labels):
        """
        Make a log-loss term based on the labels.

        Args:
          actual_classes: a Tensor representing a batch of ground truth
            labels.

        Returns:
          A 0-D Tensor representing the mean loss.
        """
        numeric_labels = tf.cast(labels, self._logits.dtype)
        raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=numeric_labels,
                                                      logits=self._logits)
        return tf.reduce_mean(raw)

class CategoryTagger(ImageTagger):
    """
    A tagger for video categories.
    """
    @property
    def class_labels(self):
        return all_categories()

def all_categories():
    """
    Get all the video categories for tagging.
    """
    return [
        "60FPS",
        "Amateur",
        "Anal",
        "Arab",
        "Asian",
        "BBW",
        "Babe",
        "Babysitter",
        "Behind The Scenes",
        "Big Ass",
        "Big Dick",
        "Big Tits",
        "Bisexual Male",
        "Blonde",
        "Blowjob",
        "Bondage",
        "Brazilian",
        "British",
        "Brunette",
        "Bukkake",
        "Cartoon",
        "Casting",
        "Celebrity",
        "College",
        "Compilation",
        "Cosplay",
        "Creampie",
        "Cuckold",
        "Cumshot",
        "Czech",
        "Described Video",
        "Double Penetration",
        "Ebony",
        "Euro",
        "Exclusive",
        "Feet",
        "Fetish",
        "Fisting",
        "For Women",
        "French",
        "Funny",
        "Gangbang",
        "Gay",
        "German",
        "HD Porn",
        "Handjob",
        "Hardcore",
        "Hentai",
        "Indian",
        "Interactive",
        "Interracial",
        "Italian",
        "Japanese",
        "Korean",
        "Latina",
        "Lesbian",
        "Live Cams",
        "MILF",
        "Massage",
        "Masturbation",
        "Mature",
        "Music",
        "Old/Young",
        "Orgy",
        "POV",
        "Parody",
        "Party",
        "Pissing",
        "Pornstar",
        "Public",
        "Pussy Licking",
        "Reality",
        "Red Head",
        "Rough Sex",
        "Russian",
        "SFW",
        "School",
        "Shemale",
        "Small Tits",
        "Smoking",
        "Solo Male",
        "Squirt",
        "Striptease",
        "Teen",
        "Threesome",
        "Toys",
        "Uniforms",
        "Verified Amateurs",
        "Verified Models",
        "Vintage",
        "Virtual Reality",
        "Webcam"
    ]
