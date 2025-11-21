import binascii
import typing as t
from abc import ABC, abstractmethod

import faiss
import numpy

from tx_extension_clip.config import BITS_IN_CLIP
from tx_extension_clip.utils.uint import int64_to_uint64, uint64_to_int64
from tx_extension_clip.utils.distance import similarity_to_distance

CLIP_HASH_TYPE = t.Union[str, bytes]


def _to_python_int(value: t.Any) -> int:
    """Convert numpy integer types to Python int (for Hamming distances)."""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _to_python_float(value: t.Any) -> float:
    """Convert numpy float types to Python float (for cosine similarity/distance)."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class CLIPFloatHashIndex(ABC):
    """Abstract base class for CLIP float vector indices. Uses cosine similarity (returns float)."""

    @abstractmethod
    def __init__(
        self,
        vectors: t.Iterable[t.Tuple[str, int]] = (),
        dimension: int = 512,
    ):
        pass

    @abstractmethod
    def add(self, vectors: t.Iterable[t.Tuple[str, int]]):
        pass

    @abstractmethod
    def search_top_k(
        self,
        queries: t.Sequence[str],
        k: int,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, float]]]:
        """Returns (id, vector_hex, distance) tuples."""
        pass

    @abstractmethod
    def search_threshold(
        self,
        queries: t.Sequence[str],
        threshold: float,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, float]]]:
        """Returns (id, vector_hex, distance) tuples where distance <= threshold."""
        pass

    @abstractmethod
    def vector_at(self, idx: int) -> numpy.ndarray:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class CLIPHashIndex(ABC):
    """Abstract base class for binary CLIP hash indices. Uses Hamming distance (returns int)."""
    
    @abstractmethod
    def __init__(self, faiss_index: faiss.IndexBinary) -> None:
        self.faiss_index = faiss_index
        super().__init__()

    @abstractmethod
    def hash_at(self, idx: int) -> str:
        """
        Returns the hash located at the given index. The index order is determined by the initial order of hashes used to
        create this index.
        """
        pass

    @abstractmethod
    def add(self, hashes: t.Iterable[CLIP_HASH_TYPE], custom_ids: t.Iterable[int]):
        """
        Adds hashes and their custom ids to the CLIP index.
        """
        pass

    def search(
        self,
        queries: t.Sequence[CLIP_HASH_TYPE],
        threshhold: int,
        return_as_ids: bool = False,
    ):
        """
        Searches this index for CLIP hashes within the index that are no more than the threshold away from the query hashes by
        hamming distance.

        Parameters
        ----------
        queries: sequence of CLIP Hashes
            The CLIP hashes to query against the index
        threshold: int
            Threshold value to use for this search. The hamming distance between the result hashes and the related query will
            be no more than the threshold value. i.e., hamming_dist(q_i,r_i_j) <= threshold.
        return_as_ids: boolean
            whether the return values should be the index ids for the matching items. Defaults to false.

        Returns
        -------
        sequence of matches per query
            For each query provided in queries, the returned sequence will contain a sequence of matches within the index
            that were within threshold hamming distance of that query. These matches will either be a hexstring of the hash
            by default, or the index ids of the matches if `return_as_ids` is True. The inner sequences may be empty in the
            case of no hashes within the index. The same CLIP hash may also appear in more than one inner sequence if it
            matches multiple query hashes.

            For example the hash "000000000000000000000000000000000000000000000000000000000000FFFF" would match both
            "00000000000000000000000000000000000000000000000000000000FFFFFFFF" and
            "0000000000000000000000000000000000000000000000000000000000000000" for a threshold of 16. Thus it would appear in
            the entry for both the hashes if they were both in the queries list.
        """
        query_vectors = [
            numpy.frombuffer(binascii.unhexlify(q), dtype=numpy.uint8) for q in queries
        ]
        qs = numpy.array(query_vectors)
        limits, _, I = self.faiss_index.range_search(qs, threshhold + 1)

        if return_as_ids:
            # for custom ids, we understood them initially as uint64 numbers and then coerced them internally to be signed
            # int64s, so we need to reverse this before returning them back to the caller. For non custom ids, this will
            # effectively return the same result
            output_fn: t.Callable[[int], t.Any] = int64_to_uint64
        else:
            output_fn = self.hash_at

        return [
            [output_fn(idx.item()) for idx in I[limits[i] : limits[i + 1]]]
            for i in range(len(query_vectors))
        ]

    def search_top_k(
        self,
        queries: t.Sequence[CLIP_HASH_TYPE],
        k: int,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, int]]]:
        """
        Search method that returns a mapping from query_str => (id, hash, distance) for the top k matches.
        It progressively increases the search breadth (`nflip`) to ensure k matches are found.
        """
        query_vectors = [
            numpy.frombuffer(binascii.unhexlify(q), dtype=numpy.uint8) for q in queries
        ]
        qs = numpy.array(query_vectors)

        output_fn: t.Callable[[int], t.Any] = int64_to_uint64
        result = {}

        max_nflip = self.mih_index.b // 2

        for i, query in enumerate(queries):
            q_vector = qs[i : i + 1]
            distances, I = None, None

            # Progressively increase search breadth to find at least k matches
            for nflip_val in range(max_nflip + 1):
                self.mih_index.nflip = nflip_val
                current_distances, current_I = self.faiss_index.search(q_vector, k)

                distances, I = current_distances, current_I

                if I is not None:
                    valid_matches = sum(1 for match_id in I[0] if match_id >= 0)
                    if valid_matches == k:
                        break

            match_tuples = []
            if I is not None:
                for j in range(k):
                    match_id = I[0][j]
                    if match_id < 0:
                        continue
                    dist = _to_python_int(distances[0][j])
                    match_tuples.append(
                        (output_fn(match_id), self.hash_at(match_id), dist)
                    )
            result[query] = match_tuples

        return result

    def search_with_distance_in_result(
        self,
        queries: t.Sequence[str],
        threshhold: int,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, int]]]:
        """
        Search method that return a mapping from query_str =>  (id, hash, distance)

        This implementation is the same as `search` above however instead of returning just the sequence of matches
        per query it returns a mapping from query strings to a list of matched hashes (or ids) and distances

        e.g.
        result = {
            "000000000000000000000000000000000000000000000000000000000000FFFF": [
                (12345678901, "00000000000000000000000000000000000000000000000000000000FFFFFFFF", 16)
            ]
        }
        """

        query_vectors = []
        try:
            for q in queries:
                query_vectors.append(numpy.frombuffer(binascii.unhexlify(q), dtype=numpy.uint8))
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"Invalid hex string in queries: {e}")
        qs = numpy.array(query_vectors)
        limits, similarities, I = self.faiss_index.range_search(qs, threshhold + 1)

        # for custom ids, we understood them initially as uint64 numbers and then coerced them internally to be signed
        # int64s, so we need to reverse this before returning them back to the caller. For non custom ids, this will
        # effectively return the same result
        output_fn: t.Callable[[int], t.Any] = int64_to_uint64

        result = {}
        for i, query in enumerate(queries):
            match_tuples = []
            matches = [idx.item() for idx in I[limits[i] : limits[i + 1]]]
            distances = [idx for idx in similarities[limits[i] : limits[i + 1]]]
            for match, distance in zip(matches, distances):
                # (Id, Hash, Distance)
                distance = _to_python_int(distance)
                match_tuples.append((output_fn(match), self.hash_at(match), distance))
            result[query] = match_tuples
        return result

    def __getstate__(self):
        data = faiss.serialize_index_binary(self.faiss_index)
        return data

    def __setstate__(self, data):
        self.faiss_index = faiss.deserialize_index_binary(data)


class CLIPFlatHashIndex(CLIPHashIndex):
    """
    Wrapper around an faiss binary index for use with searching for similar CLIP hashes

    The "flat" variant uses an exhaustive search approach that may use less memory than other approaches and may be more
    performant when using larger thresholds for CLIP similarity.
    """

    def __init__(self):
        faiss_index = faiss.IndexBinaryIDMap2(
            faiss.index_binary_factory(BITS_IN_CLIP, "BFlat")
        )
        super().__init__(faiss_index)

    def add(self, hashes: t.Iterable[CLIP_HASH_TYPE], custom_ids: t.Iterable[int]):
        """
        Parameters
        ----------
        hashes: sequence of CLIP Hashes
            The CLIP hashes to create the index with
        custom_ids: sequence of custom ids for the CLIP Hashes
            Sequence of custom id values to use for the CLIP hashes for any
            method relating to indexes (e.g., hash_at). If provided, the nth item in
            custom_ids will be used as the id for the nth hash in hashes. If not provided
            then the ids for the hashes will be assumed to be their respective index
            in hashes (i.e., the nth hash would have id n, starting from 0).
        """
        hash_bytes = [binascii.unhexlify(hash) for hash in hashes]
        vectors = list(
            map(lambda h: numpy.frombuffer(h, dtype=numpy.uint8), hash_bytes)
        )
        i64_ids = list(map(uint64_to_int64, custom_ids))
        self.faiss_index.add_with_ids(numpy.array(vectors), numpy.array(i64_ids))

    def hash_at(self, idx: int) -> str:
        i64_id = uint64_to_int64(idx)
        vector = self.faiss_index.reconstruct(i64_id)
        return binascii.hexlify(vector.tobytes()).decode()


class CLIPMultiHashIndex(CLIPHashIndex):
    """
    Wrapper around an faiss binary index for use with searching for similar CLIP hashes

    The "multi" variant uses an the Multi-Index Hashing searching technique employed by faiss's
    IndexBinaryMultiHash binary index.

    Properties:
    nhash: int (optional)
    Optional number of hashmaps for the underlaying faiss index to use for
    the Multi-Index Hashing lookups.
    """

    def __init__(self, nhash: int = 16):
        bits_per_hashmap = BITS_IN_CLIP // nhash
        faiss_index = faiss.IndexBinaryIDMap2(
            faiss.IndexBinaryMultiHash(BITS_IN_CLIP, nhash, bits_per_hashmap)
        )
        super().__init__(faiss_index)
        self.__construct_index_rev_map()

    def add(
        self,
        hashes: t.Iterable[CLIP_HASH_TYPE],
        custom_ids: t.Iterable[int],
    ):
        """
        Parameters
        ----------
        hashes: sequence of CLIP Hashes
            The CLIP hashes to create the index with
        custom_ids: sequence of custom ids for the CLIP Hashes
            Sequence of custom id values to use for the CLIP hashes for any
            method relating to indexes (e.g., hash_at). If provided, the nth item in
            custom_ids will be used as the id for the nth hash in hashes. If not provided
            then the ids for the hashes will be assumed to be their respective index
            in hashes (i.e., the nth hash would have id n, starting from 0).

        Returns
        -------
        a CLIPMultiHashIndex of these hashes
        """
        hash_bytes = [binascii.unhexlify(hash) for hash in hashes]
        vectors = list(
            map(lambda h: numpy.frombuffer(h, dtype=numpy.uint8), hash_bytes)
        )
        i64_ids = list(map(uint64_to_int64, custom_ids))
        self.faiss_index.add_with_ids(numpy.array(vectors), numpy.array(i64_ids))
        self.__construct_index_rev_map()

    @property
    def mih_index(self):
        """
        Convenience accessor for the underlaying faiss.IndexBinaryMultiHash index regardless of if it is wrapped in an ID
        map or not.
        """
        if hasattr(self.faiss_index, "index"):
            return faiss.downcast_IndexBinary(self.faiss_index.index)
        return self.faiss_index

    def search(
        self,
        queries: t.Sequence[CLIP_HASH_TYPE],
        threshhold: int,
        return_as_ids: bool = False,
    ):
        self.mih_index.nflip = threshhold // self.mih_index.nhash
        return super().search(queries, threshhold, return_as_ids)

    def search_with_distance_in_result(
        self,
        queries: t.Sequence[str],
        threshhold: int,
    ):
        self.mih_index.nflip = threshhold // self.mih_index.nhash
        return super().search_with_distance_in_result(queries, threshhold)

    def hash_at(self, idx: int) -> str:
        i64_id = uint64_to_int64(idx)
        if self.index_rev_map:
            index_id = self.index_rev_map[i64_id]
        else:
            index_id = i64_id
        vector = self.mih_index.storage.reconstruct(index_id)
        return binascii.hexlify(vector.tobytes()).decode()

    def __construct_index_rev_map(self):
        """
        Workaround method for creating an in-memory lookup mapping custom ids to internal index id representations. The
        rev_map property provided in faiss.IndexBinaryIDMap2 has no accessible `at` or other index lookup methods in swig
        and the implementation of `reconstruct` in faiss.IndexBinaryIDMap2 requires the underlaying index to directly
        support `reconstruct`, which faiss.IndexBinaryMultiHash does not. Thus this workaround is needed until either the
        values in the faiss.IndexBinaryIDMap2 rev_map can be accessed directly or faiss.IndexBinaryMultiHash is directly
        supports `reconstruct` calls.
        """
        if hasattr(self.faiss_index, "id_map"):
            id_map = self.faiss_index.id_map
            self.index_rev_map = {id_map.at(i): i for i in range(id_map.size())}
        else:
            self.index_rev_map = None

    def __setstate__(self, data):
        super().__setstate__(data)
        self.__construct_index_rev_map()


class CLIPFloatVectorIndex(CLIPFloatHashIndex):
    """FAISS float vector index using IndexFlatIP for cosine similarity. 
    
    Requires normalized vectors (IndexFlatIP computes inner product = cosine similarity only for normalized vectors).
    Normalization is verified when adding vectors to the index.
    """

    def __init__(
        self,
        vectors: t.Iterable[t.Tuple[str, int]] = (),
        dimension: int = 512,
    ):
        self.dimension = dimension
        self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.id_to_vector: t.Dict[int, numpy.ndarray] = {}
        self.add(vectors)

    def add(self, vectors: t.Iterable[t.Tuple[str, int]]):
        """Adds normalized vectors to the index (required for IndexFlatIP cosine similarity)."""
        vector_list = []
        id_list = []
        for vec_str, custom_id in vectors:
            vec = numpy.frombuffer(binascii.unhexlify(vec_str), dtype=numpy.float32)
            # Verify vectors are normalized (IndexFlatIP requires normalized vectors for cosine similarity)
            norm = numpy.linalg.norm(vec)
            if not numpy.isclose(norm, 1.0, rtol=1e-5):
                raise ValueError(f"Vector must be normalized for cosine similarity (norm={norm})")
            vector_list.append(vec)
            id_list.append(custom_id)
            self.id_to_vector[custom_id] = vec

        if not vector_list:
            return

        vectors_array = numpy.array(vector_list, dtype=numpy.float32)
        ids_array = numpy.array(id_list, dtype=numpy.int64)

        self.faiss_index.add_with_ids(vectors_array, ids_array)

    def search_top_k(
        self,
        queries: t.Sequence[str],
        k: int,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, float]]]:
        """Search for top k matches. Assumes query vectors are normalized.
        
        Returns (id, vector_hex, distance) tuples where distance is cosine distance
        in range [0, 2]. For normalized vectors, IndexFlatIP returns cosine similarity
        in range [-1, 1], which we convert to distance = 1.0 - similarity.
        """
        if len(queries) == 0:
            return {}
        
        query_vectors = [
            numpy.frombuffer(binascii.unhexlify(q), dtype=numpy.float32) for q in queries
        ]
        queries_array = numpy.array(query_vectors, dtype=numpy.float32)
        similarities, indices = self.faiss_index.search(queries_array, k)
        
        result = {}
        for i, query in enumerate(queries):
            matches = []
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0:
                    similarity = _to_python_float(similarities[i][j])
                    # Convert similarity to distance using common utility function
                    # For cosine similarity in [-1, 1], distance ranges from [0, 2]
                    distance = similarity_to_distance(similarity)
                    vector = self.id_to_vector[idx]
                    vector_hex = binascii.hexlify(vector.tobytes()).decode()
                    matches.append((int(idx), vector_hex, distance))
            result[query] = matches
        
        return result

    def search_threshold(
        self,
        queries: t.Sequence[str],
        threshold: float,
    ) -> t.Dict[str, t.List[t.Tuple[int, str, float]]]:
        """Search within distance threshold. Assumes query vectors are normalized.
        
        Args:
            threshold: Maximum distance. Returns items with distance <= threshold.
        """
        if len(queries) == 0:
            return {}
        
        # Convert max distance to min similarity: distance = 1.0 - similarity
        similarity_threshold = 1.0 - threshold
        
        # For threshold = 0.0, add small epsilon to handle floating point precision
        epsilon = 1e-5 if threshold == 0.0 else 0.0
        faiss_threshold = similarity_threshold - epsilon
        
        query_vectors = [
            numpy.frombuffer(binascii.unhexlify(q), dtype=numpy.float32) for q in queries
        ]
        queries_array = numpy.array(query_vectors, dtype=numpy.float32)
        lims, similarities, indices = self.faiss_index.range_search(
            queries_array, faiss_threshold
        )

        result = {}
        for i, query in enumerate(queries):
            matches = []
            start_idx = lims[i]
            end_idx = lims[i + 1]
            for j in range(start_idx, end_idx):
                idx = int(indices[j])
                similarity = _to_python_float(similarities[j])
                if similarity >= (similarity_threshold - epsilon):
                    distance = similarity_to_distance(similarity)
                    if distance <= threshold or (threshold == 0.0 and numpy.isclose(distance, 0.0, atol=1e-5)):
                        vector = self.id_to_vector[idx]
                        vector_hex = binascii.hexlify(vector.tobytes()).decode()
                        matches.append((idx, vector_hex, distance))
            result[query] = matches
        
        return result

    def vector_at(self, idx: int) -> numpy.ndarray:
        return self.id_to_vector[idx]

    def __len__(self) -> int:
        return self.faiss_index.ntotal

    def __getstate__(self):
        faiss_data = faiss.serialize_index(self.faiss_index)
        return {
            'dimension': self.dimension,
            'faiss_data': faiss_data,
            'id_to_vector': self.id_to_vector,
        }

    def __setstate__(self, state):
        self.dimension = state['dimension']
        self.faiss_index = faiss.deserialize_index(state['faiss_data'])
        self.id_to_vector = state['id_to_vector']
