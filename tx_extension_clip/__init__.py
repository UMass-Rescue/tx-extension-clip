import sys
import os
from unittest.mock import MagicMock

# This file is run by the test discoverer before any tests are run.

# Mock heavy dependencies for the test environment
if os.environ.get("TX_CLIP_TEST_ENV", "false").lower() == "true":
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
