"""
temporal.py
===========
Temporal buffer — stores per-person feature sequences across frames.

Each person has an independent ring buffer of fixed size (SEQUENCE_LENGTH).
The module is completely stateless with respect to camera/YOLO; it only
holds numpy arrays.

Usage::

    buffer = TemporalBuffer(sequence_length=20)
    buffer.add(person_id=1, feature_vector=features)
    if buffer.is_ready(person_id=1):
        seq = buffer.get_sequence(person_id=1)   # shape (20, FEATURE_DIM)
    buffer.clear(person_id=1)
    buffer.remove_stale(active_ids={1, 2})
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Set

import numpy as np

from ai.features import FEATURE_DIM

logger = logging.getLogger(__name__)


class TemporalBuffer:
    """
    Per-person temporal feature buffer implemented as a ring-buffer (deque).

    Parameters
    ----------
    sequence_length : int
        Number of frames to retain per person. When the buffer is full,
        the oldest frame is automatically discarded.
    """

    def __init__(self, sequence_length: int = 20):
        if sequence_length < 2:
            raise ValueError("sequence_length must be ≥ 2")
        self._seq_len = sequence_length
        # person_id → deque of np.ndarray shape (FEATURE_DIM,)
        self._buffers: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add(self, person_id: int, feature_vector: np.ndarray) -> None:
        """
        Append a feature vector for a given person.

        If the buffer for that person does not exist it is created.
        Silently ignores vectors of wrong shape (logs a warning).
        """
        fv = np.asarray(feature_vector, dtype=np.float32).flatten()
        if fv.shape[0] != FEATURE_DIM:
            logger.warning(
                "TemporalBuffer.add: expected feature dim %d, got %d – "
                "padding/truncating for person %d",
                FEATURE_DIM,
                fv.shape[0],
                person_id,
            )
            # Pad or truncate to expected size
            padded = np.zeros(FEATURE_DIM, dtype=np.float32)
            n = min(fv.shape[0], FEATURE_DIM)
            padded[:n] = fv[:n]
            fv = padded

        if person_id not in self._buffers:
            self._buffers[person_id] = deque(maxlen=self._seq_len)

        self._buffers[person_id].append(fv)

    def clear(self, person_id: int) -> None:
        """Remove all frames for a person without deleting their buffer slot."""
        if person_id in self._buffers:
            self._buffers[person_id].clear()

    def remove(self, person_id: int) -> None:
        """Completely remove the buffer for a person."""
        self._buffers.pop(person_id, None)

    def remove_stale(self, active_ids: Set[int]) -> None:
        """
        Remove buffers for persons who are no longer being tracked.
        Call once per frame with the set of person IDs detected in
        the current frame.
        """
        stale = [pid for pid in self._buffers if pid not in active_ids]
        for pid in stale:
            self.remove(pid)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def is_ready(self, person_id: int) -> bool:
        """Return True when the buffer has accumulated sequence_length frames."""
        buf = self._buffers.get(person_id)
        return buf is not None and len(buf) >= self._seq_len

    def current_length(self, person_id: int) -> int:
        """Number of frames currently stored for a person."""
        buf = self._buffers.get(person_id)
        return len(buf) if buf is not None else 0

    def get_sequence(self, person_id: int) -> Optional[np.ndarray]:
        """
        Return the most recent `sequence_length` feature vectors as an
        array of shape (sequence_length, FEATURE_DIM).

        Returns None if the person has no data or fewer frames than
        sequence_length.
        """
        buf = self._buffers.get(person_id)
        if buf is None or len(buf) == 0:
            return None
        # Return all frames if not yet full, or the full window
        seq = np.stack(list(buf), axis=0)   # (N, FEATURE_DIM)
        return seq

    def get_partial_sequence(self, person_id: int) -> Optional[np.ndarray]:
        """
        Return whatever frames are available for a person (may be fewer
        than sequence_length). Useful for fallback rule-based logic that
        only needs a few recent frames.
        """
        return self.get_sequence(person_id)

    def get_latest(self, person_id: int) -> Optional[np.ndarray]:
        """Return only the most recent feature vector, or None."""
        buf = self._buffers.get(person_id)
        if not buf:
            return None
        return buf[-1].copy()

    def get_recent(self, person_id: int, n: int) -> Optional[np.ndarray]:
        """
        Return the most recent `n` feature vectors.
        Returns None if fewer than `n` frames are available.
        """
        buf = self._buffers.get(person_id)
        if buf is None or len(buf) < n:
            return None
        frames = list(buf)[-n:]
        return np.stack(frames, axis=0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def sequence_length(self) -> int:
        return self._seq_len

    @property
    def tracked_ids(self) -> Set[int]:
        return set(self._buffers.keys())

    def __repr__(self) -> str:
        summary = {pid: len(buf) for pid, buf in self._buffers.items()}
        return f"TemporalBuffer(seq_len={self._seq_len}, persons={summary})"
