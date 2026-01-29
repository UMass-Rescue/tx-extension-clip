"""
Implementation of SignalTypeIndex abstraction for CLIP by wrapping `matcher`
"""

import logging
import typing as t

logger = logging.getLogger(__name__)

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
    CLIP_HNSW_M,
    CLIP_HNSW_EF_CONSTRUCTION,
    CLIP_HNSW_EF_SEARCH,
)
from tx_extension_clip.matcher import (
    CLIPFlatHashIndex,
    CLIPHashIndex,
    CLIPFloatVectorFlatIndex,
    CLIPFloatVectorHNSWIndex,
    CLIPMultiHashIndex,
)

CLIPIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithIntDistance, IndexT]
CLIPFloatIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]


class CLIPIndex(SignalTypeIndex[IndexT]):
    """Binary hash index wrapper using CLIPMultiHashIndex. Returns Hamming distance as int."""

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
        self, results: t.Dict[str, t.List[t.Tuple[int, str, int]]]
    ) -> t.List[CLIPIndexMatch[IndexT]]:
        """Process results from binary hash matcher (Hamming distances as int)."""
        matches = []
        for result_list in results.values():
            for id, _, distance in result_list:
                matches.append(
                    IndexMatchUntyped(
                        SignalSimilarityInfoWithIntDistance(distance),
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
    """Binary hash index wrapper using CLIPFlatHashIndex with higher threshold. Returns Hamming distance as int."""

    @classmethod
    def get_match_threshold(cls):
        return CLIP_FLAT_HASH_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> CLIPHashIndex:
        return CLIPFlatHashIndex()


class CLIPFloatIndexBase(SignalTypeIndex[IndexT]):
    """Base class for float vector index wrappers. Returns cosine distance as float.
    
    Provides shared implementation for both flat (exact) and HNSW (approximate) indices.
    """

    @classmethod
    def get_match_threshold(cls) -> float:
        return CLIP_FLOAT_SIMILARITY_THRESHOLD

    def __init__(
        self,
        entries: t.Iterable[t.Tuple[str, IndexT]],
        vector_index,
    ) -> None:
        """Initialize with a pre-configured matcher index instance.
        
        Args:
            entries: Initial entries to add
            vector_index: Pre-configured CLIPFloatVectorIndex or CLIPHNSWVectorIndex instance
        """
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[str, IndexT]] = list(entries)
        self.index = vector_index

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
        return self._process_query_results(results)

    def query_threshold(
        self, hash: str, threshold: t.Optional[float] = None
    ) -> t.Sequence[IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]]:
        """Find all matches within a given similarity threshold."""
        if threshold is None:
            threshold = self.get_match_threshold()
        results = self.index.search_threshold([hash], threshold)
        return self._process_query_results(results)

    def _process_query_results(
        self, results: t.Dict[str, t.List[t.Tuple[int, str, float]]]
    ) -> t.List[IndexMatchUntyped[SignalSimilarityInfoWithFloatDistance, IndexT]]:
        """Process results from float vector matcher (returns distance)."""
        matches = []
        for hash, result_list in results.items():
            for id, _, distance in result_list:
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


class CLIPFloatFlatIndex(CLIPFloatIndexBase):
    """Float vector index wrapper using flat (exact) search."""

    def __init__(self, entries: t.Iterable[t.Tuple[str, IndexT]] = ()) -> None:
        entry_list = list(entries)
        vector_index = CLIPFloatVectorFlatIndex(
            vectors=[(s, i) for i, (s, _) in enumerate(entry_list)],
            dimension=BITS_IN_CLIP,
        )
        super().__init__(entry_list, vector_index)
        logger.info("CLIP_SIGNAL_INDEX_TYPE: CLIPFloatFlatIndex (exact/flat search wrapper)")


class CLIPFloatHNSWIndex(CLIPFloatIndexBase):
    """Float vector index wrapper using HNSW (approximate) search.
    
    HNSW (Hierarchical Navigable Small World) provides fast approximate nearest neighbor search,
    suitable for large-scale datasets where exact search becomes too slow.
    
    Trade-offs:
    - Much faster search than CLIPFloatFlatIndex (especially for large datasets)
    - Slightly less accurate (approximate results)
    - Uses more memory due to HNSW graph structure
    - Configurable accuracy/speed trade-offs via M, ef_construction, and ef_search parameters
    """

    def __init__(
        self,
        entries: t.Iterable[t.Tuple[str, IndexT]] = (),
        M: int = CLIP_HNSW_M,
        ef_construction: int = CLIP_HNSW_EF_CONSTRUCTION,
        ef_search: int = CLIP_HNSW_EF_SEARCH,
    ) -> None:
        """
        Initialize HNSW index with configurable parameters.
        
        Args:
            entries: Initial entries to add to the index
            M: Number of connections per layer (default 32, higher = better accuracy, more memory)
            ef_construction: Size of candidate list during construction (default 200, higher = better quality, slower build)
            ef_search: Size of candidate list during search (default 128, higher = better recall, slower search)
        """
        entry_list = list(entries)
        vector_index = CLIPFloatVectorHNSWIndex(
            vectors=[(s, i) for i, (s, _) in enumerate(entry_list)],
            dimension=BITS_IN_CLIP,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        super().__init__(entry_list, vector_index)
        logger.info(f"CLIP_SIGNAL_INDEX_TYPE: CLIPFloatHNSWIndex (approximate/hnsw search wrapper, M={M}, ef_construction={ef_construction}, ef_search={ef_search})")
