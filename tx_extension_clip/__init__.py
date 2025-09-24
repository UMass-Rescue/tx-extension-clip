import sys
from unittest.mock import MagicMock

# This file is run by the test discoverer before any tests are run.

# Mock heavy dependencies for the test environment
if "unittest" in sys.modules:
    sys.modules["torch"] = MagicMock()
    sys.modules["open_clip"] = MagicMock()
    sys.modules["open_clip.factory"] = MagicMock()
    sys.modules["torchvision"] = MagicMock()
    sys.modules["torchvision.transforms"] = MagicMock()

from tx_extension_clip.manifest import CLIPExtensionManifest
from tx_extension_clip.signal import CLIPSignal

TX_MANIFEST = CLIPExtensionManifest(
    signal_types=(CLIPSignal,),
)
