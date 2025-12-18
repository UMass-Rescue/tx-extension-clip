from tx_extension_clip.manifest import CLIPExtensionManifest
from tx_extension_clip.signal import CLIPFloatSignal, CLIPSignal, CLIPHNSWSignal

TX_MANIFEST = CLIPExtensionManifest(
    signal_types=(
        CLIPSignal,
        CLIPFloatSignal,
        CLIPHNSWSignal,
    ),
)
