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

from tx_extension_clip.matcher import (
    CLIPFlatHashIndex,
    CLIPHashIndex,
    CLIPMultiHashIndex,
)
from tx_extension_clip.config import BITS_IN_CLIP

CLIP_CONFIDENT_MATCH_THRESHOLD = 0.01
CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD = 0.02

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
        return CLIP_CONFIDENT_MATCH_THRESHOLD

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

        # query takes a signal hash but index supports batch queries hence [hash]
        results = self.index.search_with_distance_in_result(
            [hash], self.get_match_threshold()
        )

        matches = []
        for id, _, distance in results[hash]:
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithIntDistance(int(distance)),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def _convert_threshold_to_int(self, threshold: t.Union[float, int]) -> int:
        """
        Convert a cosine-distance threshold to an integer Hamming threshold for FAISS.
        
        Parameters
        ----------
        threshold: float or int
            If float in [0.0, 1.0], interpreted as cosine distance with a maximum of 1.0.
            If int, assumed to already be a Hamming-distance threshold in [0, BITS_IN_CLIP].
            
        Returns
        -------
        int
            Integer Hamming threshold value for FAISS (0..BITS_IN_CLIP)

        Notes
        -----
        - Cosine distance d in [0,1] is mapped to a Hamming limit H = round(d * BITS_IN_CLIP).
        - FAISS binary indexes report distances in Hamming space. Tests that assert on distance
          should compare FAISS-returned distances to this integer value, not to the raw cosine d.
        """
        # Float thresholds are cosine-distance in [0,1]
        # Map to an integer Hamming radius using a conservative scale aligned
        # with the default number of MIH sub-hashes (nhash=16).
        clamped = max(0.0, min(1.0, float(threshold)))
        per_subhash_bits = BITS_IN_CLIP // 16  # default nhash in CLIPMultiHashIndex
        return int(round(clamped * per_subhash_bits))

    def query_threshold(self, hash: str, threshold: float) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for all matches below a distance threshold using FAISS range_search.
        
        Parameters
        ----------
        hash: str
            The CLIP hash to query against the index
        threshold: float
            Cosine-distance threshold in [0,1]. Internally converted to a Hamming threshold
            for FAISS. All matches with Hamming distance <= converted threshold are returned.
            
        Returns
        -------
        sequence of matches
            All matches within the converted Hamming threshold, sorted by Hamming distance
            (closest first). Cosine and Hamming distances are correlated but not identical.
        """
        # Convert float threshold to integer for FAISS if needed
        threshold_int = self._convert_threshold_to_int(threshold)

        # Use FAISS range_search via search_with_distance_in_result
        results = self.index.search_with_distance_in_result([hash], threshold_int)
        
        matches = []
        for id, _, distance in results[hash]:
            matches.append(
                IndexMatchUntyped(
                    # Keep FAISS-reported distance as float for parity with search_* APIs
                    SignalSimilarityInfoWithIntDistance(distance),
                    self.local_id_to_entry[id][1],
                )
            )
        
        # Sort by distance (closest first)
        matches.sort(key=lambda x: x.similarity_info.distance)
        return matches

    def query_topk(self, hash: str, k: int) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for top k matches using FAISS search.
        
        Parameters
        ----------
        hash: str
            The CLIP hash to query against the index
        k: int
            The number of top matches to return
            
        Returns
        -------
        sequence of matches
            Top k matches sorted by distance (closest first)
        """
        # Use FAISS search for top-k (more efficient than range_search for this use case)
        results = self.index.search_topk([hash], k)
        
        matches = []
        for id, _, distance in results[hash]:
            matches.append(
                IndexMatchUntyped(
                    # Keep FAISS-reported distance as float for parity with search_* APIs
                    SignalSimilarityInfoWithIntDistance(distance),
                    self.local_id_to_entry[id][1],
                )
            )
        
        # Results are already sorted by distance from FAISS search
        return matches

    def serialize(self, fout: t.BinaryIO) -> None:
        """Serialize the CLIPIndex to a binary stream"""
        import pickle
        # Use pickle for now - may need custom serialization if FAISS issues arise
        pickle.dump(self, fout)

    @classmethod
    def deserialize(cls: t.Type["CLIPIndex"], fin: t.BinaryIO) -> "CLIPIndex":
        """Deserialize a CLIPIndex from a binary stream"""
        import pickle
        return pickle.load(fin)

    @classmethod
    def build(cls: t.Type["CLIPIndex"], entries: t.Iterable[t.Tuple[str, IndexT]]) -> "CLIPIndex":
        """Build a CLIPIndex from entries"""
        ret = cls()
        ret.add_all(entries)
        return ret

    def __getstate__(self) -> dict:
        """Get state for pickling - needed because FAISS indexes require special serialization"""
        return {
            'local_id_to_entry': self.local_id_to_entry,
            'index_data': self.index.__getstate__(),
            'index_type': type(self.index).__name__,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickling - needed to reconstruct the correct FAISS index type"""
        self.local_id_to_entry = state['local_id_to_entry']
        
        # Recreate the appropriate index type
        index_type = state['index_type']
        if index_type == 'CLIPMultiHashIndex':
            self.index = self._get_empty_index()  # Uses the class's default
        elif index_type == 'CLIPFlatHashIndex':
            from tx_extension_clip.matcher import CLIPFlatHashIndex
            self.index = CLIPFlatHashIndex()
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Restore the FAISS index data
        self.index.__setstate__(state['index_data'])

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
        return CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> CLIPHashIndex:
        return CLIPFlatHashIndex()

    @classmethod
    def build(cls: t.Type["CLIPFlatIndex"], entries: t.Iterable[t.Tuple[str, IndexT]]) -> "CLIPFlatIndex":
        """Build a CLIPFlatIndex from entries"""
        ret = cls()
        ret.add_all(entries)
        return ret
