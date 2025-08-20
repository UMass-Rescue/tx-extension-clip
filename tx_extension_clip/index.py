"""
Implementation of SignalTypeIndex abstraction for CLIP by wrapping `matcher`
"""

import typing as t

from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithSingleDistance,
    SignalTypeIndex,
)
from threatexchange.signal_type.index import T as IndexT

from tx_extension_clip.matcher import (
    CLIPFlatHashIndex,
    CLIPHashIndex,
    CLIPMultiHashIndex,
)

CLIP_CONFIDENT_MATCH_THRESHOLD = 0.01
CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD = 0.02

def _hamming_to_cosine_distance(hamming_distance: float) -> float:
    """
    Convert hamming distance to approximate cosine distance for display.
    
    Uses the inverse of the relationship in _cosine_to_hamming_threshold:
    hamming = 512 * arccos(cosine_similarity) / π
    Therefore: cosine_distance = 1 - cos(hamming * π / 512)
    
    Args:
        hamming_distance: Hamming distance from FAISS (0-512)
        
    Returns:
        Approximate cosine distance (0.0-2.0)
    """
    import math
    
    # Clamp hamming distance to valid range
    hamming_distance = max(0, min(hamming_distance, 512))
    
    # Apply inverse transformation
    angle = hamming_distance * math.pi / 512
    cosine_similarity = math.cos(angle)
    cosine_distance = 1.0 - cosine_similarity
    
    # Clamp to valid cosine distance range
    return max(0.0, min(2.0, cosine_distance))


def _cosine_to_hamming_threshold(cosine_threshold: float) -> int:
    """
    Convert cosine distance threshold to hamming distance threshold for binary quantized CLIP embeddings.
    
    With proper binary quantization, there's a theoretical relationship between cosine similarity
    and hamming distance for normalized vectors:
    
    For sign-based quantization:
    P(bit_i_differs) = arccos(cosine_similarity) / π
    Expected hamming distance = dimensions * arccos(cosine_similarity) / π
    
    Args:
        cosine_threshold: Cosine distance threshold (0.0-1.0)
        
    Returns:
        Hamming distance threshold (integer)
    """
    import math
    
    # Convert cosine distance to cosine similarity
    cosine_similarity = 1.0 - cosine_threshold
    
    # Clamp to valid range [-1, 1]
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    
    # Theoretical relationship for sign-based binary quantization
    # Expected hamming distance = d * arccos(cosine_similarity) / π
    # where d is the dimension (512 for CLIP)
    expected_hamming = 512 * math.acos(cosine_similarity) / math.pi
    
    # Add some tolerance (±20%) to account for variance
    tolerance_factor = 1.2
    hamming_threshold = int(expected_hamming * tolerance_factor)
    
    # Clamp to reasonable bounds
    return max(1, min(hamming_threshold, 512))

CLIPIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithSingleDistance[float], IndexT]


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
        # Convert cosine threshold to hamming threshold for FAISS binary index
        hamming_threshold = _cosine_to_hamming_threshold(self.get_match_threshold())
        results = self.index.search_with_distance_in_result(
            [hash], hamming_threshold
        )

        matches = []
        for id, _, distance in results[hash]:
            # Convert hamming distance to cosine distance for meaningful display
            cosine_dist = _hamming_to_cosine_distance(float(distance))
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](cosine_dist),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def query_threshold(self, hash: str, threshold: float) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for all matches below the specified distance threshold.
        
        Args:
            hash: The signal hash to query
            threshold: Maximum cosine distance threshold for matches (converted to hamming distance)
            
        Returns:
            All matches with hamming distance <= converted threshold
        """
        # Convert cosine distance threshold to hamming distance threshold
        # Note: FAISS binary index uses hamming distance (integer), not cosine distance
        hamming_threshold = _cosine_to_hamming_threshold(threshold)
        results = self.index.search_with_distance_in_result([hash], hamming_threshold)
        
        matches = []
        for id, _, distance in results[hash]:
            # Convert hamming distance to cosine distance for meaningful display
            cosine_dist = _hamming_to_cosine_distance(float(distance))
            matches.append(
                IndexMatchUntyped(
                    SignalSimilarityInfoWithSingleDistance[float](cosine_dist),
                    self.local_id_to_entry[id][1],
                )
            )
        return matches

    def query_topk(self, hash: str, k: int) -> t.Sequence[CLIPIndexMatch[IndexT]]:
        """
        Query for the top k closest matches.
        
        Args:
            hash: The signal hash to query
            k: Number of top matches to return
            
        Returns:
            Top k matches ordered by distance (closest first)
        """
        # Convert hash to query vector - consistent with matcher.py pattern
        import binascii
        import numpy
        
        query_vector = numpy.frombuffer(binascii.unhexlify(hash), dtype=numpy.uint8)
        qs = numpy.array([query_vector])
        
        # Use FAISS search to get top k matches
        distances, indices = self.index.faiss_index.search(qs, k)
        
        matches = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            
            # Skip invalid indices (FAISS returns -1 for missing results)
            if idx >= 0 and idx < len(self.local_id_to_entry):
                # Convert hamming distance to cosine distance for meaningful display
                cosine_dist = _hamming_to_cosine_distance(float(distance))
                matches.append(
                    IndexMatchUntyped(
                        SignalSimilarityInfoWithSingleDistance[float](cosine_dist),
                        self.local_id_to_entry[idx][1],
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
        return CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> CLIPHashIndex:
        return CLIPFlatHashIndex()
