#!/usr/bin/env python

from torchio.data import GridSampler
from ...utils import TorchioTestCase


class TestGridSampler(TorchioTestCase):
    """Tests for `GridSampler`."""

    def test_locations(self):
        sampler = GridSampler(self.sample, (5, 20, 20), (2, 4, 6))
        fixture = [
            [0, 0, 0, 5, 20, 20],
            [0, 0, 10, 5, 20, 30],
            [3, 0, 0, 8, 20, 20],
            [3, 0, 10, 8, 20, 30],
            [5, 0, 0, 10, 20, 20],
            [5, 0, 10, 10, 20, 30],
        ]
        locations = sampler.locations.tolist()
        self.assertEqual(locations, fixture)

    def test_large_patch(self):
        with self.assertRaises(ValueError):
            GridSampler(self.sample, (5, 21, 5), (1, 2, 3))

    def test_large_overlap(self):
        with self.assertRaises(ValueError):
            GridSampler(self.sample, (5, 20, 5), (2, 4, 6))

    def test_single_location(self):
        sampler = GridSampler(self.sample, (10, 20, 30), 0)
        fixture = [[0, 0, 0, 10, 20, 30]]
        self.assertEqual(sampler.locations.tolist(), fixture)
