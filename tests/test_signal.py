import pathlib
import unittest
import binascii
from unittest.mock import patch, MagicMock

from tests.test_utils import MOCKED_MODULES

patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.signal import CLIPSignal, CLIPFloatSignal
from tx_extension_clip.index import CLIPFloatFlatIndex, CLIPFloatHNSWIndex

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

    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file(self, mock_hasher):
        class MockClipOutput:
            def serialize(self):
                return 'mocked_file_hash'

        mock_hasher.hash_from_file.return_value = MockClipOutput()

        path = pathlib.Path("/fake/image.jpg")
        result = CLIPSignal.hash_from_file(path)
        self.assertEqual(result, 'mocked_file_hash')
        mock_hasher.hash_from_file.assert_called_once_with(path)

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list(self, mock_hasher, mock_image):
        class MockClipOutput:
            def __init__(self, value):
                self._value = value

            def serialize(self):
                return self._value

        mock_images = [MagicMock(), MagicMock()]
        mock_image.open.side_effect = mock_images
        mock_hasher.hash_from_image_list.return_value = [
            MockClipOutput('hash_a'),
            MockClipOutput('hash_b'),
        ]

        paths = [pathlib.Path("/fake/a.jpg"), pathlib.Path("/fake/b.jpg")]
        result = CLIPSignal.hash_from_file_list(paths)
        self.assertEqual(result, ['hash_a', 'hash_b'])
        self.assertEqual(mock_image.open.call_count, 2)
        mock_hasher.hash_from_image_list.assert_called_once_with(mock_images)
        for img in mock_images:
            img.close.assert_called_once()

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list_empty(self, mock_hasher, mock_image):
        result = CLIPSignal.hash_from_file_list([])
        self.assertEqual(result, [])
        mock_image.open.assert_not_called()
        mock_hasher.hash_from_image_list.assert_not_called()

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list_batching(self, mock_hasher, mock_image):
        """Paths exceeding batch_size should be split across multiple hasher calls."""
        class MockClipOutput:
            def __init__(self, value):
                self._value = value

            def serialize(self):
                return self._value

        num_paths = 5
        paths = [pathlib.Path(f"/fake/{i}.jpg") for i in range(num_paths)]
        mock_image.open.side_effect = [MagicMock() for _ in range(num_paths)]
        mock_hasher.hash_from_image_list.side_effect = [
            [MockClipOutput(f"hash_{i}") for i in range(3)],
            [MockClipOutput(f"hash_{i}") for i in range(3, 5)],
        ]

        result = CLIPSignal.hash_from_file_list(paths, batch_size=3)

        self.assertEqual(result, [f"hash_{i}" for i in range(5)])
        self.assertEqual(mock_hasher.hash_from_image_list.call_count, 2)


class TestCLIPFloatSignalIndexSelection(unittest.TestCase):
    def test_default_index_cls_is_hnsw(self):
        """Test that HNSW is the default (CLIP_DISABLE_HNSW_INDEX=False by default)."""
        self.assertIs(CLIPFloatSignal.get_index_cls(), CLIPFloatHNSWIndex)

    @patch("tx_extension_clip.signal.CLIP_DISABLE_HNSW_INDEX", True)
    def test_flat_index_cls_when_hnsw_disabled(self):
        """Test that Flat index is used when HNSW is disabled."""
        self.assertIs(CLIPFloatSignal.get_index_cls(), CLIPFloatFlatIndex)


class TestCLIPFloatSignalFileHashing(unittest.TestCase):
    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file(self, mock_hasher, mock_image):
        mock_image.open.return_value = MagicMock()
        mock_hasher.get_float_vector_strs_from_image_list.return_value = ['float_hash_a']

        path = pathlib.Path("/fake/image.jpg")
        result = CLIPFloatSignal.hash_from_file(path)

        self.assertEqual(result, 'float_hash_a')
        mock_image.open.assert_called_once_with(path)
        mock_hasher.get_float_vector_strs_from_image_list.assert_called_once()

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list(self, mock_hasher, mock_image):
        mock_images = [MagicMock(), MagicMock()]
        mock_image.open.side_effect = mock_images
        mock_hasher.get_float_vector_strs_from_image_list.return_value = [
            'float_hash_a',
            'float_hash_b',
        ]

        paths = [pathlib.Path("/fake/a.jpg"), pathlib.Path("/fake/b.jpg")]
        result = CLIPFloatSignal.hash_from_file_list(paths)

        self.assertEqual(result, ['float_hash_a', 'float_hash_b'])
        self.assertEqual(mock_image.open.call_count, 2)
        mock_hasher.get_float_vector_strs_from_image_list.assert_called_once_with(mock_images)
        for img in mock_images:
            img.close.assert_called_once()

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list_empty(self, mock_hasher, mock_image):
        result = CLIPFloatSignal.hash_from_file_list([])

        self.assertEqual(result, [])
        mock_image.open.assert_not_called()
        mock_hasher.get_float_vector_strs_from_image_list.assert_not_called()

    @patch("tx_extension_clip.signal.Image")
    @patch("tx_extension_clip.signal.CLIP_HASHER")
    def test_hash_from_file_list_batching(self, mock_hasher, mock_image):
        """Paths exceeding batch_size should be split across multiple hasher calls."""
        num_paths = 5
        paths = [pathlib.Path(f"/fake/{i}.jpg") for i in range(num_paths)]
        mock_image.open.side_effect = [MagicMock() for _ in range(num_paths)]
        mock_hasher.get_float_vector_strs_from_image_list.side_effect = [
            [f"hash_{i}" for i in range(3)],
            [f"hash_{i}" for i in range(3, 5)],
        ]

        result = CLIPFloatSignal.hash_from_file_list(paths, batch_size=3)

        self.assertEqual(result, [f"hash_{i}" for i in range(5)])
        self.assertEqual(mock_hasher.get_float_vector_strs_from_image_list.call_count, 2)


if __name__ == '__main__':
    unittest.main()
