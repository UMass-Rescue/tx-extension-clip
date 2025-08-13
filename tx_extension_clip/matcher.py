import typing as t
from abc import ABC, abstractmethod

import faiss
import numpy

from tx_extension_clip.utils.uint import int64_to_uint64, uint64_to_int64


class CLIPFloatIndex(ABC):
    """
    Abstract base class for CLIP float-based indexes that support cosine similarity.
    These indexes work with the original float embeddings rather than binary hashes.
    """
    
    @abstractmethod
    def __init__(self, faiss_index: faiss.Index) -> None:
        self.faiss_index = faiss_index
        super().__init__()

    @abstractmethod
    def embedding_at(self, idx: int) -> numpy.ndarray:
        """
        Returns the embedding located at the given index.
        """
        pass

    @abstractmethod
    def add(self, embeddings: t.Iterable[numpy.ndarray], custom_ids: t.Iterable[int]):
        """
        Adds embeddings and their custom ids to the CLIP index.
        """
        pass

    def search_with_distance_in_result(
        self,
        queries: t.Sequence[numpy.ndarray],
        threshold: float,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """
        Search method that returns a mapping from query_idx => (id, embedding, distance)
        
        For cosine similarity, the distance is 1 - cosine_similarity, so lower values
        indicate more similar embeddings.
        """
        qs = numpy.array(queries)
        limits, distances, I = self.faiss_index.range_search(qs, threshold)

        # for custom ids, we understood them initially as uint64 numbers and then coerced them internally to be signed
        # int64s, so we need to reverse this before returning them back to the caller
        output_fn: t.Callable[[int], t.Any] = int64_to_uint64

        result = {}
        for i, query in enumerate(queries):
            match_tuples = []
            matches = [idx.item() for idx in I[limits[i] : limits[i + 1]]]
            query_distances = [dist for dist in distances[limits[i] : limits[i + 1]]]
            for match, distance in zip(matches, query_distances):
                # (Id, Embedding, Distance)
                match_tuples.append((output_fn(match), self.embedding_at(match), distance))
            result[i] = match_tuples
        return result

    def search_topk(
        self,
        queries: t.Sequence[numpy.ndarray],
        k: int,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """
        Search method that returns the top k closest matches for each query.
        
        For cosine similarity, returns the k most similar embeddings.
        """
        qs = numpy.array(queries)
        distances, I = self.faiss_index.search(qs, k)

        # for custom ids, we understood them initially as uint64 numbers and then coerced them internally to be signed
        # int64s, so we need to reverse this before returning them back to the caller
        output_fn: t.Callable[[int], t.Any] = int64_to_uint64

        result = {}
        for i, query in enumerate(queries):
            match_tuples = []
            matches = [idx.item() for idx in I[i]]
            query_distances = [dist for dist in distances[i]]
            
            # Filter out invalid indices (FAISS returns -1 for invalid results)
            for match, distance in zip(matches, query_distances):
                if match != -1:  # FAISS returns -1 for invalid indices
                    # (Id, Embedding, Distance)
                    match_tuples.append((output_fn(match), self.embedding_at(match), distance))
            
            result[i] = match_tuples
        return result

    def __getstate__(self):
        data = faiss.serialize_index(self.faiss_index)
        return data

    def __setstate__(self, data):
        self.faiss_index = faiss.deserialize_index(data)


class CLIPFlatFloatIndex(CLIPFloatIndex):
    """
    Wrapper around a FAISS flat float index for use with CLIP embeddings.
    
    This index uses exhaustive search and supports cosine similarity.
    The embeddings should be normalized for optimal cosine similarity performance.
    """
    
    def __init__(self, dimension: int = 512):
        # Create a flat index with cosine similarity
        faiss_index = faiss.IndexIDMap2(
            faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
        )
        super().__init__(faiss_index)
        self.dimension = dimension

    def add(self, embeddings: t.Iterable[numpy.ndarray], custom_ids: t.Iterable[int]):
        """
        Parameters
        ----------
        embeddings: sequence of CLIP embeddings
            The CLIP embeddings to add to the index. Should be normalized for cosine similarity.
        custom_ids: sequence of custom ids for the embeddings
            Sequence of custom id values to use for the embeddings.
        """
        embedding_vectors = [numpy.array(emb, dtype=numpy.float32) for emb in embeddings]
        vectors = numpy.array(embedding_vectors)
        i64_ids = list(map(uint64_to_int64, custom_ids))
        self.faiss_index.add_with_ids(vectors, numpy.array(i64_ids))

    def embedding_at(self, idx: int) -> numpy.ndarray:
        i64_id = uint64_to_int64(idx)
        vector = self.faiss_index.reconstruct(i64_id)
        return vector


class CLIPIVFFlatFloatIndex(CLIPFloatIndex):
    """
    Wrapper around a FAISS IVF flat float index for use with CLIP embeddings.
    
    This index uses inverted file search for faster approximate search while maintaining
    good accuracy. Supports cosine similarity.
    """
    
    def __init__(self, dimension: int = 512, nlist: int = 100):
        # Create an IVF flat index with cosine similarity
        quantizer = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.IndexIDMap2(
            faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        )
        super().__init__(faiss_index)
        self.dimension = dimension
        self.nlist = nlist

    def add(self, embeddings: t.Iterable[numpy.ndarray], custom_ids: t.Iterable[int]):
        """
        Parameters
        ----------
        embeddings: sequence of CLIP embeddings
            The CLIP embeddings to add to the index. Should be normalized for cosine similarity.
        custom_ids: sequence of custom ids for the embeddings
            Sequence of custom id values to use for the embeddings.
        """
        embedding_vectors = [numpy.array(emb, dtype=numpy.float32) for emb in embeddings]
        vectors = numpy.array(embedding_vectors)
        i64_ids = list(map(uint64_to_int64, custom_ids))
        self.faiss_index.add_with_ids(vectors, numpy.array(i64_ids))

    def embedding_at(self, idx: int) -> numpy.ndarray:
        i64_id = uint64_to_int64(idx)
        vector = self.faiss_index.reconstruct(i64_id)
        return vector

    def search_with_distance_in_result(
        self,
        queries: t.Sequence[numpy.ndarray],
        threshold: float,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """
        Override to set nprobe for IVF search.
        """
        # Set nprobe for IVF search (higher values = more accurate but slower)
        self.faiss_index.nprobe = min(self.nlist, max(1, self.nlist // 10))
        return super().search_with_distance_in_result(queries, threshold)

    def search_topk(
        self,
        queries: t.Sequence[numpy.ndarray],
        k: int,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """
        Override to set nprobe for IVF search.
        """
        # Set nprobe for IVF search (higher values = more accurate but slower)
        self.faiss_index.nprobe = min(self.nlist, max(1, self.nlist // 10))
        return super().search_topk(queries, k)
