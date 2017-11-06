"""
A multi-purpose video prediction model.
"""

from .aggregate import Aggregate, aggregate_from_data
from .data import DataDir, hotspot_dataset, popularity_dataset, category_dataset
from .hotspots import HotspotPredictor
from .images import ImageNetwork
from .popularity import PopularityPredictor
from .tagging import CategoryTagger
