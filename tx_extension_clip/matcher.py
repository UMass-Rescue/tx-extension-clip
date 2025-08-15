import typing as t
from abc import ABC, abstractmethod

import faiss
import numpy

from tx_extension_clip.utils.uint import int64_to_uint64, uint64_to_int64
from tx_extension_clip.config import CLIP_IVFPQ_NLIST, CLIP_IVFPQ_M, CLIP_IVFPQ_NBITS


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


class CLIPIVFPQFloatIndex(CLIPFloatIndex):
    """
    Wrapper around a FAISS IVFPQ float index for use with CLIP embeddings.
    
    This index uses inverted file search combined with product quantization for 
    highly memory-efficient approximate search. Best for very large datasets where
    memory usage is a concern. Supports cosine similarity.
    """
    
    def __init__(self, dimension: int = 512, nlist: int = CLIP_IVFPQ_NLIST, m: int = CLIP_IVFPQ_M, nbits: int = CLIP_IVFPQ_NBITS):
        """
        Parameters
        ----------
        dimension: int
            Dimension of the CLIP embeddings (default 512)
        nlist: int  
            Number of inverted file clusters (default 1000)
        m: int
            Number of PQ sub-vectors (default 64). Must divide dimension evenly.
        nbits: int
            Bits per sub-vector (default 8). Higher = more accurate but more memory.
        """
        # Create the quantizer (coarse index for IVF)
        quantizer = faiss.IndexFlatIP(dimension)
        
        # Create IVFPQ index with cosine similarity (inner product for normalized vectors)
        faiss_index = faiss.IndexIDMap2(
            faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        )
        
        super().__init__(faiss_index)
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self._is_trained = False

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
        
        # Ensure vectors are normalized for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Train the index if not already trained
        if not self._is_trained:
            self._train_index(vectors)
        
        i64_ids = list(map(uint64_to_int64, custom_ids))
        self.faiss_index.add_with_ids(vectors, numpy.array(i64_ids))

    def _get_optimal_nlist(self, n_vectors: int) -> int:
        """
        Get optimal number of clusters based on dataset size.
        Simple rule: roughly sqrt(n_vectors), constrained by training requirements.
        """
        import math
        
        # Need at least 39 training points per cluster
        max_nlist = max(1, n_vectors // 39)
        
        # Simple rule: sqrt of dataset size
        target_nlist = max(1, int(math.sqrt(n_vectors)))
        
        return min(max_nlist, target_nlist)

    def _train_index(self, training_vectors: numpy.ndarray) -> None:
        """
        Train the IVFPQ index with the provided training vectors.
        """
        n_train: int = len(training_vectors)
        
        # Auto-configure clusters based on dataset size
        optimal_nlist: int = self._get_optimal_nlist(n_train)
        
        if optimal_nlist != self.nlist:
            # Recreate the index with optimal nlist
            quantizer = faiss.IndexFlatIP(self.dimension)
            new_faiss_index = faiss.IndexIDMap2(
                faiss.IndexIVFPQ(quantizer, self.dimension, optimal_nlist, self.m, 
                                self.nbits, faiss.METRIC_INNER_PRODUCT)
            )
            self.faiss_index = new_faiss_index
            self.nlist = optimal_nlist
        
        # Use subset for training if dataset is large
        train_subset: numpy.ndarray = training_vectors[:100000] if len(training_vectors) > 100000 else training_vectors
            
        try:
            self.faiss_index.train(train_subset)
            self._is_trained = True
        except RuntimeError as e:
            if "Number of training points" in str(e):
                # Fall back to flat index for insufficient data
                self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
                self._is_trained = True
            else:
                raise

    def embedding_at(self, idx: int) -> numpy.ndarray:
        """
        Note: IVFPQ does not support exact reconstruction. This returns an approximation.
        """
        i64_id = uint64_to_int64(idx)
        # IVFPQ only supports approximate reconstruction
        try:
            vector = self.faiss_index.reconstruct(i64_id)
            return vector
        except RuntimeError:
            # If reconstruction fails, IVFPQ doesn't store exact vectors
            raise NotImplementedError(
                "IVFPQ index does not support exact vector reconstruction. "
                "Consider using IVFFlat if you need exact reconstruction."
            )

    def _prepare_search(self, queries: t.Sequence[numpy.ndarray]) -> numpy.ndarray:
        """Prepare queries for IVFPQ search."""
        qs = numpy.array(queries, dtype=numpy.float32)
        faiss.normalize_L2(qs)
        self.faiss_index.nprobe = min(self.nlist, max(1, self.nlist // 10))
        return qs

    def search_with_distance_in_result(
        self,
        queries: t.Sequence[numpy.ndarray],
        threshold: float,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """Override to set nprobe and normalize queries."""
        qs = self._prepare_search(queries)
        limits, distances, I = self.faiss_index.range_search(qs, threshold)
        
        return self._process_results(qs, I, distances, limits=limits)

    def search_topk(
        self,
        queries: t.Sequence[numpy.ndarray],
        k: int,
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """Override to set nprobe and normalize queries."""
        qs = self._prepare_search(queries)
        distances, I = self.faiss_index.search(qs, k)
        
        return self._process_results(qs, I, distances)

    def _process_results(
        self, 
        queries: numpy.ndarray, 
        indices: numpy.ndarray, 
        distances: numpy.ndarray, 
        limits: t.Optional[numpy.ndarray] = None
    ) -> t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]]:
        """Process FAISS results into the expected format."""
        output_fn: t.Callable[[int], int] = int64_to_uint64
        result: t.Dict[int, t.List[t.Tuple[int, numpy.ndarray, numpy.float32]]] = {}
        
        for i, query in enumerate(queries):
            match_tuples: t.List[t.Tuple[int, numpy.ndarray, numpy.float32]] = []
            
            if limits is not None:
                # Range search results
                matches: t.List[int] = [idx.item() for idx in indices[limits[i] : limits[i + 1]]]
                query_distances: t.List[float] = [dist for dist in distances[limits[i] : limits[i + 1]]]
            else:
                # Top-k search results
                matches: t.List[int] = [idx.item() for idx in indices[i]]
                query_distances: t.List[float] = [dist for dist in distances[i]]
            
            for match, distance in zip(matches, query_distances):
                if match != -1:  # Filter invalid indices
                    # For IVFPQ, use the query as embedding approximation
                    match_tuples.append((output_fn(match), query, distance))
                    
            result[i] = match_tuples
        return result
