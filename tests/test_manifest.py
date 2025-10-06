import unittest
from unittest.mock import patch, MagicMock

from tests.test_utils import MOCKED_MODULES

# Mock heavy dependencies before importing tx_extension_clip modules
patch.dict("sys.modules", MOCKED_MODULES).start()

from tx_extension_clip.manifest import CLIPExtensionManifest

class TestCLIPExtensionManifest(unittest.TestCase):
    @patch('tx_extension_clip.manifest.CLIPHasher')
    def test_bootstrap(self, mock_clip_hasher):
        # Create a mock instance of the hasher
        mock_hasher_instance = MagicMock()
        mock_clip_hasher.return_value = mock_hasher_instance

        # Call the bootstrap method
        CLIPExtensionManifest.bootstrap()

        # Assert that the hasher was initialized and the model download was triggered
        mock_clip_hasher.assert_called_once()
        mock_hasher_instance.init_model_and_transforms.assert_called_once()

if __name__ == '__main__':
    unittest.main()
