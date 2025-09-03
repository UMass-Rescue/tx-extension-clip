"""Configuration for CLIP Threat Exchange Extension."""
from tx_extension_clip.hasher import CLIPHasher

# Note that the embeddings must match the model used to generate the hashes.
# This metadata must be included in the database.
# Hamming distance threshold for binary CLIP hash comparison (number of differing bits)
CLIP_MULTI_HASH_MATCH_THRESHOLD: int = 92   # Conservative threshold, scaled to ~12% of 768 bits
CLIP_FLAT_HASH_MATCH_THRESHOLD: int = 115   # Slightly higher threshold, scaled to ~15% of 768 bits
CLIP_DISTANCE_THRESHOLD: int = (
    CLIP_MULTI_HASH_MATCH_THRESHOLD  # The default for one-to-one comparisons
)
CLIP_NORMALIZED: bool = True
OPEN_CLIP_MODEL_NAME: str = "ViT-L-14"
OPEN_CLIP_PRETRAINED: str = "laion2b_s32b_b82k"
BITS_IN_CLIP: int = 768  # 768 binary features from quantized CLIP embeddings

CLIP_HASHER: CLIPHasher = CLIPHasher(
    model_name=OPEN_CLIP_MODEL_NAME,
    pretrained=OPEN_CLIP_PRETRAINED,
    normalized=CLIP_NORMALIZED,
)
