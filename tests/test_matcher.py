import unittest
from unittest.mock import patch

from tests.test_utils import MOCKED_MODULES

patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.matcher import CLIPFlatHashIndex, CLIPMultiHashIndex

class TestCLIPMatchers(unittest.TestCase):
    def setUp(self):
        self.hashes = [
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "ffffffffffffffffffff000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ]
        self.ids = list(range(len(self.hashes)))

    def _test_index_search(self, index):
        query = self.hashes[0]
        results = index.search([query], 0)
      
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[0][0], query)
        results_with_dist = index.search_with_distance_in_result([query], 0)
        self.assertIn(query, results_with_dist)
        self.assertEqual(len(results_with_dist[query]), 1)
        match_id, match_hash, match_dist = results_with_dist[query][0]
        self.assertEqual(match_id, self.ids[0])
        self.assertEqual(match_hash, query)
        self.assertEqual(match_dist, 0.0)

    def test_flat_hash_index(self):
        index = CLIPFlatHashIndex()
        index.add(self.hashes, self.ids)
        for i, h in enumerate(self.hashes):
            self.assertEqual(index.hash_at(i), h)
        self._test_index_search(index)


    def test_multi_hash_index(self):
        index = CLIPMultiHashIndex()
        index.add(self.hashes, self.ids)
        for i, h in enumerate(self.hashes):
            self.assertEqual(index.hash_at(i), h)
        self._test_index_search(index)

        # test search top k
        query = self.hashes[0]
        results_top_k = index.search_top_k([query], 1)
        self.assertIn(query, results_top_k)
        self.assertEqual(len(results_top_k[query]), 1)
        match_id_k, match_hash_k, match_dist_k = results_top_k[query][0]
        self.assertEqual(match_id_k, self.ids[0])
        self.assertEqual(match_hash_k, query)
        self.assertEqual(match_dist_k, 0.0)


if __name__ == '__main__':
    unittest.main()
