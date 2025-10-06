import unittest
import binascii
import numpy as np
from unittest.mock import MagicMock, patch

from tests.test_utils import MOCKED_MODULES

# Mock heavy dependencies before importing tx_extension_clip modules
patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.index import CLIPIndex, CLIPFlatIndex


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


if __name__ == "__main__":
    unittest.main()
