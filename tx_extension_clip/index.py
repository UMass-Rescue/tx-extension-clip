"""
Implementation of SignalTypeIndex abstraction for CLIP by wrapping `matcher`
"""

import typing as t

from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithIntDistance,
    SignalSimilarityInfoWithSingleDistance,
    SignalTypeIndex,
)
from threatexchange.signal_type.index import T as IndexT

# Create float distance type (similar to how ThreatExchange defines IntDistance)
SignalSimilarityInfoWithFloatDistance = SignalSimilarityInfoWithSingleDistance[float]

from tx_extension_clip.config import (
    BITS_IN_CLIP,
    CLIP_FLAT_HASH_MATCH_THRESHOLD,
    CLIP_FLOAT_SIMILARITY_THRESHOLD,
    CLIP_MULTI_HASH_MATCH_THRESHOLD,
)
from tx_extension_clip.matcher import (
    CLIPFlatHashIndex,
    CLIPHashIndex,
    CLIPFloatVectorIndex,
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
        self, hash: str, threshold: int
    ) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index, up to the given threshold.
        """
        results = self.index.search_with_distance_in_result([hash], threshold)
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


class CLIPFloatIndex(SignalTypeIndex[IndexT]):
    """
    Wrapper around CLIPIndexFlatIP for float vector cosine similarity.
    """

    @classmethod
    def get_match_threshold(cls) -> float:
        return CLIP_FLOAT_SIMILARITY_THRESHOLD

    def __init__(self, entries: t.Iterable[t.Tuple[str, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[str, IndexT]] = list(entries)
        self.index: CLIPFloatVectorIndex = CLIPFloatVectorIndex(
            vectors=[(s, i) for i, (s, _) in enumerate(self.local_id_to_entry)],
            dimension=BITS_IN_CLIP,
        )

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def query(
        self, hash: str
    ) -> t.Sequence[IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]]:
        return self.query_threshold(hash)

    def query_top_k(
        self, hash: str, k: int
    ) -> t.Sequence[IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]]:
        """Find the top k closest matches to a given hash."""
        results = self.index.search_top_k([hash], k)

        matches = []
        if hash in results:
            for id, _, similarity in results[hash]:
                distance = 1.0 - similarity
                matches.append(
                    IndexMatchUntyped(
                        SignalSimilarityInfoWithFloatDistance(distance),
                        self.local_id_to_entry[id][1],
                    )
                )
        return matches

    def query_threshold(
        self, hash: str, threshold: t.Optional[float] = None
    ) -> t.Sequence[IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]]:
        """Find all matches within a given similarity threshold."""
        if threshold is None:
            threshold = self.get_match_threshold()
        results = self.index.search_threshold([hash], threshold)

        matches = []
        if hash in results:
            for id, _, similarity in results[hash]:
                distance = 1.0 - similarity
                matches.append(
                    IndexMatchUntyped(
                        SignalSimilarityInfoWithFloatDistance(distance),
                        self.local_id_to_entry[id][1],
                    )
                )
        return matches

    def add(self, signal_str: str, entry: IndexT) -> None:
        self.add_all(((signal_str, entry),))

    def add_all(self, entries: t.Iterable[t.Tuple[str, IndexT]]) -> None:
        start = len(self)
        entry_list = list(entries)
        if not entry_list:
            return
        self.local_id_to_entry.extend(entry_list)
        self.index.add(
            [(s, i + start) for i, (s, _) in enumerate(entry_list)],
        )
