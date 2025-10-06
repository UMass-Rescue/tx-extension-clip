import unittest
import numpy as np
from unittest.mock import patch

from tests.test_utils import MOCKED_MODULES

patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.utils.distance import hamming_distance, cosine_distance

class TestHammingDistance(unittest.TestCase):
    def test_identical_hashes(self):
        hash1 = b'\x00\xff\x00\xff'
        hash2 = b'\x00\xff\x00\xff'
        self.assertEqual(hamming_distance(hash1, hash2), 0)

    def test_completely_different_hashes(self):
        hash1 = b'\x00\x00\x00\x00'
        hash2 = b'\xff\xff\xff\xff'
        self.assertEqual(hamming_distance(hash1, hash2), 32)

    def test_known_distance(self):
        hash1 = b'\x01'  # 00000001
        hash2 = b'\x03'  # 00000011
        self.assertEqual(hamming_distance(hash1, hash2), 1)

    def test_mixed_hashes(self):
        hash1 = b'\x12\x34\x56\x78'
        hash2 = b'\x87\x65\x43\x21'
        self.assertEqual(hamming_distance(hash1, hash2), 14)

    def test_empty_hashes(self):
        hash1 = b''
        hash2 = b''
        self.assertEqual(hamming_distance(hash1, hash2), 0)

    def test_different_length_hashes_throws_error(self):
        hash1 = b'\x01'
        hash2 = b'\x01\x02'
        with self.assertRaises(ValueError):
            hamming_distance(hash1, hash2)


class TestCosineDistance(unittest.TestCase):
    def test_identical_vectors(self):
        v1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(cosine_distance(v1, v2), 0.0)

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([-1.0, -1.0, -1.0])
        self.assertAlmostEqual(cosine_distance(v1, v2), 2.0)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_distance(v1, v2), 1.0)

    def test_known_distance(self):
        v1 = np.array([3.0, 4.0])
        v2 = np.array([5.0, 12.0])
        expected = 2 / 65
        self.assertAlmostEqual(cosine_distance(v1, v2), expected)

    def test_zero_vector_vs_non_zero(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(cosine_distance(v1, v2)))

    def test_zero_vector_vs_zero_vector(self):
        v1 = np.array([0.0, 0.0, 0.0])
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(cosine_distance(v1, v1)))
    
    def test_different_length_vectors(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            cosine_distance(v1, v2)


if __name__ == '__main__':
    unittest.main()
