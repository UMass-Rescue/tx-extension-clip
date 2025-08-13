"""Configuration for CLIP Threat Exchange Extension."""
from tx_extension_clip.hasher import CLIPHasher

# Note that the embeddings must match the model used to generate the hashes.
# This metadata must be included in the database.
CLIP_NORMALIZED: bool = True
OPEN_CLIP_MODEL_NAME: str = "xlm-roberta-base-ViT-B-32"
OPEN_CLIP_PRETRAINED: str = "laion5b_s13b_b90k"

# Index thresholds for cosine similarity
# These thresholds are cosine distance values (lower = more similar)
CLIP_CONFIDENT_MATCH_THRESHOLD: float = 0.1  # 1 - 0.9 = 0.1 for 90% similarity
CLIP_FLAT_CONFIDENT_MATCH_THRESHOLD: float = 0.2  # 1 - 0.8 = 0.2 for 80% similarity

# Legacy threshold for backward compatibility (updated for cosine distance)
CLIP_DISTANCE_THRESHOLD: float = 0.1  # Updated for cosine distance (90% similarity)

CLIP_HASHER: CLIPHasher = CLIPHasher(
    model_name=OPEN_CLIP_MODEL_NAME,
    pretrained=OPEN_CLIP_PRETRAINED,
    normalized=CLIP_NORMALIZED,
)
