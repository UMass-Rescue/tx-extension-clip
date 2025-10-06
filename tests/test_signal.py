import unittest
import binascii
from unittest.mock import patch, MagicMock

# Mock heavy dependencies before importing tx_extension_clip modules
# Use patch.dict to install mocks in sys.modules
patch.dict('sys.modules', {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'open_clip': MagicMock(),
    'open_clip.factory': MagicMock(),
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
}).start()

from tx_extension_clip.signal import CLIPSignal

class TestCLIPSignal(unittest.TestCase):
    def test_validate_signal_str(self):
        # A valid hash is 128 characters hex string.
        valid_hash = "a" * 128
        self.assertEqual(CLIPSignal.validate_signal_str(valid_hash), valid_hash)

        with self.assertRaises(ValueError):
            CLIPSignal.validate_signal_str("a" * 127)

        with self.assertRaises(ValueError):
            CLIPSignal.validate_signal_str("a" * 129)

    @patch("tx_extension_clip.signal.hamming_distance")
    def test_compare_hash(self, mock_hamming_distance):
        hash1 = "0" * 128
        hash2 = "f" * 128
        threshold = 10

        # Test no match
        mock_hamming_distance.return_value = 512
        result = CLIPSignal.compare_hash(hash1, hash2, threshold)
        self.assertFalse(result.match)
        self.assertEqual(result.distance.distance, 512)
        mock_hamming_distance.assert_called_with(
            binascii.unhexlify(hash1.encode("ascii")),
            binascii.unhexlify(hash2.encode("ascii")),
        )

        # Test exact match
        mock_hamming_distance.return_value = 0
        result_identical = CLIPSignal.compare_hash(hash1, hash1, threshold)
        self.assertTrue(result_identical.match)
        self.assertEqual(result_identical.distance.distance, 0)

        # Test match within threshold
        hash3 = "1" + "0" * 127
        mock_hamming_distance.return_value = 1
        result_within = CLIPSignal.compare_hash(hash1, hash3, threshold)
        self.assertTrue(result_within.match)
        self.assertEqual(result_within.distance.distance, 1)

    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_bytes(self, mock_hasher):
        class MockClipOutput:
            def serialize(self):
                return 'mocked_hash'

        mock_hasher.hash_from_bytes.return_value = MockClipOutput()

        result = CLIPSignal.hash_from_bytes(b'some image bytes')
        self.assertEqual(result, 'mocked_hash')
        mock_hasher.hash_from_bytes.assert_called_once_with(b'some image bytes')


if __name__ == '__main__':
    unittest.main()
