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
from tx_extension_clip.signal import CLIPSignal

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

    def __init__(self, entries: t.Iterable[t.Tuple[str, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[numpy.ndarray, IndexT]] = []
        self.index = self._get_empty_index()
        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def query(self, signal_str: str) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index using a signal string.
        This is what HMA calls for normal queries.
        """
        # Convert signal string to embedding
        embedding = CLIPSignal.deserialize_embedding(signal_str)
        return self.query_embedding(embedding)

    def query_embedding(self, embedding: numpy.ndarray) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index using an embedding directly.
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

    def query_topk(self, signal_str: str, k: int) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up top k entries against the index using a signal string.
        This is what HMA calls for topk queries.
        """
        # Convert signal string to embedding
        embedding = CLIPSignal.deserialize_embedding(signal_str)
        
        results = self.index.search_topk([embedding], k)

        matches = []
        for id, _, distance in results[0]:
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def query_threshold(self, signal_str: str, threshold: float) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index with a specific threshold using a signal string.
        This is what HMA calls for threshold queries.
        """
        # Convert signal string to embedding
        embedding = CLIPSignal.deserialize_embedding(signal_str)
        
        results = self.index.search_with_distance_in_result([embedding], threshold)

        matches = []
        for id, _, distance in results[0]:
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def add(self, signal_str: str, entry: IndexT) -> None:
        """
        Add an entry to the index using a signal string.
        This is what HMA calls.
        """
        embedding = CLIPSignal.deserialize_embedding(signal_str)
        self._add_embeddings([(embedding, entry)])

    def add_all(self, entries: t.Iterable[t.Tuple[str, IndexT]]) -> None:
        """
        Add multiple entries to the index using signal strings.
        This is what HMA calls.
        """
        # Convert signal strings to embeddings
        embedding_entries = []
        for signal_str, metadata in entries:
            embedding = CLIPSignal.deserialize_embedding(signal_str)
            embedding_entries.append((embedding, metadata))
        
        self._add_embeddings(embedding_entries)

    def add_embedding(self, embedding: numpy.ndarray, entry: IndexT) -> None:
        """Add an entry using an embedding directly."""
        self._add_embeddings([(embedding, entry)])

    def _add_embeddings(self, entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]]) -> None:
        """Add multiple entries using embeddings directly."""
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
