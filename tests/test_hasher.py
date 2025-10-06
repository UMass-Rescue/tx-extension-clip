import sys
import unittest
import numpy as np
from PIL import Image
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

from tx_extension_clip.hasher import CLIPOutput, CLIPHasher
from tx_extension_clip.config import (
    OPEN_CLIP_MODEL_NAME,
    OPEN_CLIP_PRETRAINED,
    CLIP_NORMALIZED,
)


class TestCLIPHasher(unittest.TestCase):
    def setUp(self):
        self.hasher = CLIPHasher(
            model_name=OPEN_CLIP_MODEL_NAME,
            pretrained=OPEN_CLIP_PRETRAINED,
            normalized=CLIP_NORMALIZED,
        )

    def test_clip_output_serialization(self):
        clip_output = CLIPOutput(
            model_name="test_model",
            pretrained="test_pretrained",
            normalized=True,
            hash_vector=np.array([1, 2, 3, 4], dtype=np.uint8),
        )
        serialized = clip_output.serialize()
        self.assertEqual(serialized, "01020304")

        deserialized = clip_output.deserialize(serialized)
        self.assertEqual(deserialized.model_name, "test_model")
        self.assertEqual(deserialized.pretrained, "test_pretrained")
        self.assertTrue(deserialized.normalized)
        np.testing.assert_array_equal(deserialized.hash_vector, np.array([1, 2, 3, 4], dtype=np.uint8))

    def test_quantize_to_binary(self):
        float_embedding = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
        # mean is -0.05
        # binary_bits is [ True, False, True, False, True, False, True, False ]
        # which is [1,0,1,0,1,0,1,0]
        # packed is 0b10101010 = 170 = 0xAA
        binary_hash = self.hasher._quantize_to_binary(float_embedding)
        self.assertEqual(binary_hash.shape, (1,))
        self.assertEqual(binary_hash[0], 170)

    @patch("tx_extension_clip.hasher.torch")
    @patch("tx_extension_clip.hasher.open_clip.create_model_and_transforms")
    def test_hash_from_image(self, mock_create_model, mock_torch):
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)

        hasher = CLIPHasher(
            model_name=OPEN_CLIP_MODEL_NAME,
            pretrained=OPEN_CLIP_PRETRAINED,
            normalized=CLIP_NORMALIZED,
        )

        mock_numpy_array = np.ones((512,), dtype=np.float32)
        image_feature_mock = MagicMock()
        image_feature_mock.numpy.return_value = mock_numpy_array
        
        # Set up torch mocks properly
        mock_transformed_images = MagicMock()
        mock_torch.stack.return_value = mock_transformed_images
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
        mock_torch.nn.functional.normalize.side_effect = lambda x, dim: x
        
        mock_model.visual.return_value = [image_feature_mock]

        expected_hash = np.array([1, 2, 3, 4], dtype=np.uint8)
        with patch.object(
            hasher, "_quantize_to_binary", return_value=expected_hash
        ) as mock_quantize:
            image = Image.new("RGB", (224, 224), color="red")
            clip_output = hasher.hash_from_image(image)

            mock_preprocess.assert_called_once_with(image)
            mock_model.visual.assert_called_once()
            mock_quantize.assert_called_once_with(mock_numpy_array)
            np.testing.assert_array_equal(clip_output[0].hash_vector, expected_hash)


if __name__ == "__main__":
    unittest.main()
