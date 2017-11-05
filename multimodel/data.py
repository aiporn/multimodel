"""
Tools for reading data and feeding it into a model.
"""

import glob
import json
import os
import random

from scipy.interpolate import interp1d

class DataDir:
    """
    A handle on a data directory.

    The directory should contain one sub-directory per video.
    Each sub-directory contains a metadata.json file and a set of files by the
    name of thumbnail_x.png, where x is a timestamp in seconds.

    Works with the directory structure produced by:
    https://github.com/aiporn/pornhub-scraper.
    """
    def __init__(self, path):
        """
        Load meta-data about a data directory.
        """
        self._root_dir = path
        self._id_to_meta = {}
        self._id_to_path = {}
        for entry in os.listdir(path):
            dir_path = os.path.join(path, entry)
            json_path = os.path.join(dir_path, 'matadata.json')
            if (entry.startswith('.') or not os.path.isdir(dir_path) or
                    not os.path.exists(json_path)):
                continue
            with open(json_path, 'r') as json_file:
                metadata = json.load(json_file)
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
