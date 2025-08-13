"""
Implementation of SignalTypeIndex abstraction for CLIP using float-based indexes with cosine similarity
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
    CLIPFlatFloatIndex as CLIPFlatFloatIndexMatcher,
    CLIPIVFFlatFloatIndex as CLIPIVFFlatFloatIndexMatcher,
)
from tx_extension_clip.config import (
    CLIP_CONFIDENT_MATCH_THRESHOLD,
    CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD,
)

CLIPIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithSingleDistance[float], IndexT]


class CLIPFloatIndexBase(SignalTypeIndex[IndexT]):
    """
    Base class for CLIP indexes that work with float embeddings and cosine similarity.
    These indexes work with the original CLIP embeddings rather than binary hashes.
    """

    @classmethod
    def get_match_threshold(cls):
        """
        Default cosine distance threshold for confident matches.
        Distance of 0.1 corresponds to 90% cosine similarity.
        """
        return CLIP_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls):
        """Override in subclasses to return the appropriate float index"""
        raise NotImplementedError

    def __init__(self, entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[numpy.ndarray, IndexT]] = []
        self.index = self._get_empty_index()
        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def query(self, embedding: numpy.ndarray) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Look up entries against the index using cosine similarity.
        
        Parameters
        ----------
        embedding: numpy.ndarray
            The CLIP embedding to query against the index
            
        Returns
        -------
        sequence of matches
            All matches within the confident match threshold, sorted by distance
        """
        results = self.index.search_with_distance_in_result(
            [embedding], self.get_match_threshold()
        )

        matches = []
        for id, _, distance in results[0]:  # results[0] because we only have one query
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),  # Preserve float precision
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def query_threshold(self, embedding: numpy.ndarray, threshold: float) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for all matches below a cosine distance threshold.
        
        Parameters
        ----------
        embedding: numpy.ndarray
            The CLIP embedding to query against the index
        threshold: float
            Cosine distance threshold in [0,1]. Lower values mean more similar.
            A threshold of 0.1 means 90% cosine similarity.
            
        Returns
        -------
        sequence of matches
            All matches within the threshold, sorted by distance (closest first)
        """
        results = self.index.search_with_distance_in_result([embedding], threshold)
        
        matches = []
        for id, _, distance in results[0]:  # results[0] because we only have one query
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),  # Preserve float precision
                    self.local_id_to_entry[id][1],
                )
            )
        
        # Sort by distance (closest first)
        matches.sort(key=lambda x: x.similarity_info.distance)
        return matches

    def query_topk(self, embedding: numpy.ndarray, k: int) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for top k matches using cosine similarity.
        
        Parameters
        ----------
        embedding: numpy.ndarray
            The CLIP embedding to query against the index
        k: int
            The number of top matches to return
            
        Returns
        -------
        sequence of matches
            Top k matches sorted by distance (closest first)
        """
        results = self.index.search_topk([embedding], k)
        
        matches = []
        for id, _, distance in results[0]:  # results[0] because we only have one query
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](float(distance)),  # Preserve float precision
                    self.local_id_to_entry[id][1],
                )
            )
        
        # Results are already sorted by distance from FAISS search
        return matches

    def serialize(self, fout: t.BinaryIO) -> None:
        """Serialize the CLIPFloatIndexBase to a binary stream"""
        import pickle
        pickle.dump(self, fout)

    @classmethod
    def deserialize(cls: t.Type["CLIPFloatIndexBase"], fin: t.BinaryIO) -> "CLIPFloatIndexBase":
        """Deserialize a CLIPFloatIndexBase from a binary stream"""
        import pickle
        return pickle.load(fin)

    @classmethod
    def build(cls: t.Type["CLIPFloatIndexBase"], entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]]) -> "CLIPFloatIndexBase":
        """Build a CLIPFloatIndexBase from entries"""
        ret = cls()
        ret.add_all(entries)
        return ret

    def __getstate__(self) -> dict:
        """Get state for pickling"""
        return {
            'local_id_to_entry': self.local_id_to_entry,
            'index_data': self.index.__getstate__(),
            'index_type': type(self.index).__name__,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickling"""
        self.local_id_to_entry = state['local_id_to_entry']
        
        # Recreate the appropriate index type
        index_type = state['index_type']
        if index_type == 'CLIPFlatFloatIndex':
            self.index = CLIPFlatFloatIndexMatcher()
        elif index_type == 'CLIPIVFFlatFloatIndex':
            self.index = CLIPIVFFlatFloatIndexMatcher()
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Restore the FAISS index data
        self.index.__setstate__(state['index_data'])

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


class CLIPFlatFloatIndex(CLIPFloatIndexBase):
    """
    Wrapper around a FAISS flat float index for CLIP embeddings with cosine similarity.
    
    This index uses exhaustive search and provides exact results but may be slower
    for large datasets.
    """

    @classmethod
    def get_match_threshold(cls):
        return CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls):
        return CLIPFlatFloatIndexMatcher()

    @classmethod
    def build(cls: t.Type["CLIPFlatFloatIndex"], entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]]) -> "CLIPFlatFloatIndex":
        """Build a CLIPFlatFloatIndex from entries"""
        ret = cls()
        ret.add_all(entries)
        return ret


class CLIPIVFFlatFloatIndex(CLIPFloatIndexBase):
    """
    Wrapper around a FAISS IVF flat float index for CLIP embeddings with cosine similarity.
    
    This index uses inverted file search for faster approximate search while maintaining
    good accuracy. Suitable for larger datasets.
    """

    @classmethod
    def get_match_threshold(cls):
        return CLIP_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls):
        return CLIPIVFFlatFloatIndexMatcher()

    @classmethod
    def build(cls: t.Type["CLIPIVFFlatFloatIndex"], entries: t.Iterable[t.Tuple[numpy.ndarray, IndexT]]) -> "CLIPIVFFlatFloatIndex":
        """Build a CLIPIVFFlatFloatIndex from entries"""
        ret = cls()
        ret.add_all(entries)
        return ret
