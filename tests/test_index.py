import unittest
import binascii
import numpy as np
from unittest.mock import MagicMock, patch

from tests.test_utils import MOCKED_MODULES

patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.index import CLIPIndex, CLIPFlatIndex, CLIPFloatIndex, CLIPHNSWIndex


class TestCLIPIndices(unittest.TestCase):
    def setUp(self):
        self.hashes = [
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "ffffffffffffffffffff000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ]
        self.entries = [(h, i) for i, h in enumerate(self.hashes)]

    def _test_index_queries(self, index_cls):
        index = index_cls(self.entries)
        self.assertEqual(len(index), len(self.hashes))

        query_hash = self.hashes[0]
        results = index.query(query_hash)
        threshold = index_cls.get_match_threshold()

        # In our test data, hashes at indices 0, 1, and 2 are within the threshold
        # of both CLIPIndex (62) and CLIPFlatIndex (76).
        self.assertEqual(len(results), 3)

        found_metadatas = sorted([r.metadata for r in results])
        self.assertEqual(found_metadatas, [0, 1, 2])

        for result in results:
            self.assertLessEqual(result.similarity_info.distance, threshold)

    def test_clip_index_query(self):
        self._test_index_queries(CLIPIndex)

    def test_clip_flat_index_query(self):
        self._test_index_queries(CLIPFlatIndex)

    def test_clip_index_query_threshold(self):
        index = CLIPIndex()
        index.local_id_to_entry = self.entries
        query_hash = self.hashes[0]
        threshold = 10

        mock_search_result = {
            query_hash: [
                (0, self.hashes[0], 0),
                (1, self.hashes[1], 1),
                (2, self.hashes[2], 4),
            ]
        }
        index.index = MagicMock()
        index.index.search_with_distance_in_result.return_value = mock_search_result

        results = index.query_threshold(query_hash, threshold)

        index.index.search_with_distance_in_result.assert_called_once_with(
            [query_hash], threshold
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].metadata, 0)
        self.assertEqual(results[1].metadata, 1)
        self.assertEqual(results[2].metadata, 2)
        self.assertEqual(results[0].similarity_info.distance, 0)
        self.assertEqual(results[1].similarity_info.distance, 1)
        self.assertEqual(results[2].similarity_info.distance, 4)

    def test_clip_index_query_top_k(self):
        index = CLIPIndex()
        index.local_id_to_entry = self.entries
        query_hash = self.hashes[0]
        k = 2

        mock_search_result = {
            query_hash: [(0, self.hashes[0], 0), (1, self.hashes[1], 1)]
        }
        index.index = MagicMock()
        index.index.search_top_k.return_value = mock_search_result

        results = index.query_top_k(query_hash, k)

        index.index.search_top_k.assert_called_once_with([query_hash], k)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metadata, 0)
        self.assertEqual(results[1].metadata, 1)
        self.assertEqual(results[0].similarity_info.distance, 0)
        self.assertEqual(results[1].similarity_info.distance, 1)


class TestCLIPFloatIndices(unittest.TestCase):
    def setUp(self):
        """Set up test vectors (normalized float32 vectors represented as hex strings)."""
        # Create normalized random vectors for testing
        np.random.seed(42)
        vectors = []
        for i in range(4):
            vec = np.random.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            vectors.append(vec)
        
        self.vector_hashes = [binascii.hexlify(v.tobytes()).decode() for v in vectors]
        self.entries = [(h, i) for i, h in enumerate(self.vector_hashes)]

    def test_clip_float_index_query(self):
        """Test CLIPFloatIndex basic query functionality."""
        index = CLIPFloatIndex(self.entries)
        self.assertEqual(len(index), len(self.vector_hashes))

        query_hash = self.vector_hashes[0]
        results = index.query(query_hash)
        
        # Should find itself with distance ~0
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].metadata, 0)
        self.assertLess(results[0].similarity_info.distance, 0.01)

    def test_clip_float_index_query_top_k(self):
        """Test CLIPFloatIndex top-k query."""
        index = CLIPFloatIndex(self.entries)
        query_hash = self.vector_hashes[0]
        k = 2

        results = index.query_top_k(query_hash, k)
        
        self.assertEqual(len(results), k)
        # Results should be sorted by distance
        for i in range(len(results) - 1):
            self.assertLessEqual(
                results[i].similarity_info.distance,
                results[i + 1].similarity_info.distance
            )

    def test_clip_float_index_query_threshold(self):
        """Test CLIPFloatIndex threshold-based query."""
        index = CLIPFloatIndex(self.entries)
        query_hash = self.vector_hashes[0]
        threshold = 0.5

        results = index.query_threshold(query_hash, threshold)
        
        # All results should be within threshold
        for result in results:
            self.assertLessEqual(result.similarity_info.distance, threshold)

    def test_clip_hnsw_index_query(self):
        """Test CLIPHNSWIndex basic query functionality."""
        index = CLIPHNSWIndex(self.entries)
        self.assertEqual(len(index), len(self.vector_hashes))

        query_hash = self.vector_hashes[0]
        results = index.query(query_hash)
        
        # Should find itself with distance ~0
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].metadata, 0)
        self.assertLess(results[0].similarity_info.distance, 0.01)

    def test_clip_hnsw_index_query_top_k(self):
        """Test CLIPHNSWIndex top-k query."""
        index = CLIPHNSWIndex(self.entries)
        query_hash = self.vector_hashes[0]
        k = 2

        results = index.query_top_k(query_hash, k)
        
        self.assertEqual(len(results), k)
        # Results should be sorted by distance (approximate)
        for i in range(len(results) - 1):
            self.assertLessEqual(
                results[i].similarity_info.distance,
                results[i + 1].similarity_info.distance + 0.01  # Allow small tolerance for HNSW approximation
            )

    def test_clip_hnsw_index_query_threshold(self):
        """Test CLIPHNSWIndex threshold-based query."""
        index = CLIPHNSWIndex(self.entries)
        query_hash = self.vector_hashes[0]
        threshold = 0.5

        results = index.query_threshold(query_hash, threshold)
        
        # All results should be within threshold
        for result in results:
            self.assertLessEqual(result.similarity_info.distance, threshold)

    def test_clip_hnsw_index_custom_parameters(self):
        """Test CLIPHNSWIndex with custom HNSW parameters."""
        # Create index with custom parameters
        index = CLIPHNSWIndex(
            self.entries,
            M=16,
            ef_construction=100,
            ef_search=64
        )
        
        self.assertEqual(len(index), len(self.vector_hashes))
        
        query_hash = self.vector_hashes[0]
        results = index.query_top_k(query_hash, 2)
        
        # Should still work with custom parameters
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metadata, 0)

    def test_clip_hnsw_index_add_incremental(self):
        """Test incremental addition to CLIPHNSWIndex."""
        # Create empty index
        index = CLIPHNSWIndex()
        self.assertEqual(len(index), 0)
        
        # Add entries one by one
        for hash, metadata in self.entries:
            index.add(hash, metadata)
        
        self.assertEqual(len(index), len(self.vector_hashes))
        
        # Verify query still works
        query_hash = self.vector_hashes[0]
        results = index.query_top_k(query_hash, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata, 0)

    def test_clip_hnsw_vs_float_consistency(self):
        """Compare HNSW results with exact Float index for consistency."""
        float_index = CLIPFloatIndex(self.entries)
        hnsw_index = CLIPHNSWIndex(self.entries)
        
        query_hash = self.vector_hashes[0]
        k = 3
        
        float_results = float_index.query_top_k(query_hash, k)
        hnsw_results = hnsw_index.query_top_k(query_hash, k)
        
        # HNSW should find similar results (may not be identical due to approximation)
        self.assertEqual(len(float_results), k)
        self.assertEqual(len(hnsw_results), k)
        
        # The top result should be the same (the query itself)
        self.assertEqual(float_results[0].metadata, hnsw_results[0].metadata)
        
        # Distances should be close
        self.assertAlmostEqual(
            float_results[0].similarity_info.distance,
            hnsw_results[0].similarity_info.distance,
            places=2
        )


if __name__ == "__main__":
    unittest.main()
