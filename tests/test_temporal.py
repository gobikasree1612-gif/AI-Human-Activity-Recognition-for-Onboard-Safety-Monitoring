"""
test_temporal.py
================
Unit tests for the temporal buffer module.

Runnable without GPU.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.features import FEATURE_DIM
from ai.temporal import TemporalBuffer


def _rand_fv() -> np.ndarray:
    return np.random.rand(FEATURE_DIM).astype(np.float32)


class TestTemporalBufferCreation(unittest.TestCase):

    def test_create_default(self):
        buf = TemporalBuffer()
        self.assertEqual(buf.sequence_length, 20)

    def test_create_custom(self):
        buf = TemporalBuffer(sequence_length=10)
        self.assertEqual(buf.sequence_length, 10)

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            TemporalBuffer(sequence_length=1)


class TestTemporalBufferAddAndRetrieve(unittest.TestCase):

    def setUp(self):
        self.buf = TemporalBuffer(sequence_length=5)

    def test_empty_buffer_not_ready(self):
        self.assertFalse(self.buf.is_ready(person_id=1))

    def test_partial_buffer_not_ready(self):
        for _ in range(3):
            self.buf.add(1, _rand_fv())
        self.assertFalse(self.buf.is_ready(1))

    def test_full_buffer_ready(self):
        for _ in range(5):
            self.buf.add(1, _rand_fv())
        self.assertTrue(self.buf.is_ready(1))

    def test_sequence_shape_when_full(self):
        for _ in range(5):
            self.buf.add(1, _rand_fv())
        seq = self.buf.get_sequence(1)
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (5, FEATURE_DIM))

    def test_overflow_keeps_last_n(self):
        """Adding more than seq_len should keep only the most recent."""
        for i in range(10):
            fv = np.full(FEATURE_DIM, float(i), dtype=np.float32)
            self.buf.add(1, fv)
        seq = self.buf.get_sequence(1)
        # Last 5 frames should have values 5, 6, 7, 8, 9
        self.assertEqual(seq.shape[0], 5)
        np.testing.assert_allclose(seq[0, 0], 5.0)
        np.testing.assert_allclose(seq[-1, 0], 9.0)

    def test_get_latest_returns_last_frame(self):
        for i in range(3):
            fv = np.full(FEATURE_DIM, float(i), dtype=np.float32)
            self.buf.add(1, fv)
        latest = self.buf.get_latest(1)
        np.testing.assert_allclose(latest[0], 2.0)

    def test_get_latest_empty_returns_none(self):
        self.assertIsNone(self.buf.get_latest(99))

    def test_get_sequence_empty_returns_none(self):
        self.assertIsNone(self.buf.get_sequence(99))

    def test_current_length(self):
        self.assertEqual(self.buf.current_length(1), 0)
        self.buf.add(1, _rand_fv())
        self.assertEqual(self.buf.current_length(1), 1)


class TestTemporalBufferMultiplePeople(unittest.TestCase):

    def test_independent_buffers(self):
        """Person 1 and Person 2 buffers must not interfere."""
        buf = TemporalBuffer(sequence_length=5)
        fv1 = np.ones(FEATURE_DIM, dtype=np.float32)
        fv2 = np.full(FEATURE_DIM, 2.0, dtype=np.float32)

        for _ in range(5):
            buf.add(1, fv1)
            buf.add(2, fv2)

        seq1 = buf.get_sequence(1)
        seq2 = buf.get_sequence(2)

        np.testing.assert_allclose(seq1[0, 0], 1.0)
        np.testing.assert_allclose(seq2[0, 0], 2.0)

    def test_clear_one_person_leaves_other_intact(self):
        buf = TemporalBuffer(sequence_length=5)
        for _ in range(5):
            buf.add(1, _rand_fv())
            buf.add(2, _rand_fv())

        buf.clear(1)
        self.assertFalse(buf.is_ready(1))
        self.assertTrue(buf.is_ready(2))

    def test_remove_stale(self):
        buf = TemporalBuffer(sequence_length=5)
        for _ in range(5):
            buf.add(1, _rand_fv())
            buf.add(2, _rand_fv())

        # Only person 2 is now active
        buf.remove_stale(active_ids={2})
        self.assertIsNone(buf.get_sequence(1))
        self.assertIsNotNone(buf.get_sequence(2))

    def test_tracked_ids(self):
        buf = TemporalBuffer(sequence_length=5)
        buf.add(1, _rand_fv())
        buf.add(3, _rand_fv())
        self.assertEqual(buf.tracked_ids, {1, 3})


class TestTemporalBufferEdgeCases(unittest.TestCase):

    def test_wrong_dim_does_not_crash(self):
        """Feature vectors of wrong dimension should be padded/truncated safely."""
        buf = TemporalBuffer(sequence_length=3)
        bad_fv = np.ones(10, dtype=np.float32)   # Wrong dim
        buf.add(1, bad_fv)    # Should not raise
        self.assertEqual(buf.current_length(1), 1)

    def test_get_recent(self):
        buf = TemporalBuffer(sequence_length=10)
        for i in range(7):
            fv = np.full(FEATURE_DIM, float(i), dtype=np.float32)
            buf.add(1, fv)
        recent = buf.get_recent(1, 3)
        self.assertIsNotNone(recent)
        self.assertEqual(recent.shape[0], 3)
        np.testing.assert_allclose(recent[-1, 0], 6.0)

    def test_get_recent_not_enough_frames_returns_none(self):
        buf = TemporalBuffer(sequence_length=10)
        buf.add(1, _rand_fv())
        self.assertIsNone(buf.get_recent(1, 5))


if __name__ == "__main__":
    unittest.main()
