"""
Implementation of SignalTypeIndex abstraction for CLIP by wrapping `matcher`
"""

import typing as t
import numpy

from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithSingleDistance,
    SignalTypeIndex,
)
from threatexchange.signal_type.index import T as IndexT

from tx_extension_clip.matcher import (
    CLIPFlatFloatIndex,
    CLIPIVFFlatFloatIndex,
    CLIPIVFPQFloatIndex,
)
from tx_extension_clip.config import (
    CLIP_CONFIDENT_MATCH_THRESHOLD,
    CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD,
)

CLIPIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithSingleDistance[float], IndexT]


class CLIPIndex(SignalTypeIndex[IndexT]):
    """
    Wrapper around the CLIP faiss index lib using CLIPIVFPQFloatIndex
    """

    @classmethod
    def get_match_threshold(cls):
        """
        Distance should be 0.1
        Similarity should be 0.90
        """
        return CLIP_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls):
        return CLIPIVFPQFloatIndex()

    def __init__(self, entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[numpy.ndarray, IndexT]] = []
        self.index = self._get_empty_index()
        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def query(self, embedding: numpy.ndarray) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index, up to the max supported distance.
        """

        results = self.index.search_with_distance_in_result(
            [embedding], self.get_match_threshold()
        )

        matches = []
        for id, _, distance in results[0]:
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def add(self, embedding: numpy.ndarray, entry: IndexT) -> None:
        self.add_all(((embedding, entry),))

    def add_all(self, entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]]) -> None:
        start = len(self.local_id_to_entry)
        self.local_id_to_entry.extend(entries)
        if start != len(self.local_id_to_entry):
            self.index.add(
                (e[0] for e in self.local_id_to_entry[start:]),
                range(start, len(self.local_id_to_entry)),
            )


class CLIPFlatIndex(CLIPIndex):
    """
    Wrapper around the clip faiss index lib
    that uses CLIPFlatFloatIndex instead of CLIPIVFPQFloatIndex
    It also uses a high match threshold to increase recall
    possibly as the cost of precision.
    """

    @classmethod
    def get_match_threshold(cls):
        return CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls):
        return CLIPFlatFloatIndex()
