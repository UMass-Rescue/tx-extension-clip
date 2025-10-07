"""
Implementation of SignalTypeIndex abstraction for CLIP by wrapping `matcher`
"""

import typing as t

from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithIntDistance,
    SignalTypeIndex,
)
from threatexchange.signal_type.index import T as IndexT

from tx_extension_clip.config import (
    CLIP_FLAT_HASH_MATCH_THRESHOLD,
    CLIP_MULTI_HASH_MATCH_THRESHOLD,
)
from tx_extension_clip.matcher import (
    CLIPFlatHashIndex,
    CLIPHashIndex,
    CLIPMultiHashIndex,
)

CLIPIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithIntDistance, IndexT]


class CLIPIndex(SignalTypeIndex[IndexT]):
    """
    Wrapper around the CLIP faiss index lib using CLIPMultiHashIndex
    """

    @classmethod
    def get_match_threshold(cls):
        """
        Distance should be 0.01
        Similarity should be 0.990020
        """
        return CLIP_MULTI_HASH_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> CLIPHashIndex:
        return CLIPMultiHashIndex()

    def __init__(self, entries: t.Iterable[t.Tuple[str, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[str, IndexT]] = []
        self.index: CLIPHashIndex = self._get_empty_index()
        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def query(self, hash: str) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index, up to the max supported distance.
        """
        return self.query_threshold(hash, self.get_match_threshold())

    def query_threshold(
        self, hash: str, threshold: str
    ) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index, up to the given threshold.
        """
        results = self.index.search_with_distance_in_result([hash], int(threshold))
        return self._process_query_results(results)

    def query_top_k(self, hash: str, k: int) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up the top K closest entries against the index.
        """
        results = self.index.search_top_k([hash], k)
        return self._process_query_results(results)

    def _process_query_results(
        self, results: t.Dict[str, t.List[t.Tuple[int, str, float]]]
    ) -> t.List[CLIPIndexMatch[IndexT]]:
        matches = []
        for result_list in results.values():
            for id, _, distance in result_list:
                matches.append(
                    IndexMatchUntyped(
                        SignalSimilarityInfoWithIntDistance(int(distance)),
                        self.local_id_to_entry[id][1],
                    )
                )
        return matches

    def add(self, signal_str: str, entry: IndexT) -> None:
        self.add_all(((signal_str, entry),))

    def add_all(self, entries: t.Iterable[t.Tuple[str, IndexT]]) -> None:
        start = len(self.local_id_to_entry)
        self.local_id_to_entry.extend(entries)
        if start != len(self.local_id_to_entry):
            # This function signature is very silly
            self.index.add(
                (e[0] for e in self.local_id_to_entry[start:]),
                range(start, len(self.local_id_to_entry)),
            )


class CLIPFlatIndex(CLIPIndex):
    """
    Wrapper around the clip faiss index lib
    that uses CLIPFlatHashIndex instead of CLIPMultiHashIndex
    It also uses a high match threshold to increase recall
    possibly as the cost of precision.
    """

    @classmethod
    def get_match_threshold(cls):
        return CLIP_FLAT_HASH_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> CLIPHashIndex:
        return CLIPFlatHashIndex()
