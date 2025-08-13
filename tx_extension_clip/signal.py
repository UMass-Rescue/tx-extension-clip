"""
CLIP Signal Type.
"""

import typing as t

import numpy as np
from threatexchange.content_type.content_base import ContentType
from threatexchange.content_type.photo import PhotoContent
from threatexchange.signal_type import signal_base

from tx_extension_clip.config import CLIP_DISTANCE_THRESHOLD, CLIP_HASHER
from tx_extension_clip.index import CLIPFlatFloatIndex
from tx_extension_clip.utils.distance import cosine_distance


class CLIPSignal(
    signal_base.SignalType,
    signal_base.BytesHasher,
):
    """
    CLIP Signal Type.
    Article: https://arxiv.org/pdf/2103.00020.pdf
    CLIP is a neural network trained on a variety of (image, text) pairs.
    It can be used to generate image embeddings with semantic similarity, meaning
    that images with similar content will have similar embeddings.
    For example, two different images of cats will have higher cosine similarity
    than an image of a cat and an image of a tree.
    This type of hashing is robust to perceptual differences as long as the
    semantic content is the same.
    
    This implementation uses float-based indexes with cosine similarity for
    more accurate similarity measurements compared to binary hashes.
    """

    INDICATOR_TYPE: str = "HASH_CLIP"

    @classmethod
    def get_content_types(cls) -> t.List[t.Type[ContentType]]:
        return [PhotoContent]

    @classmethod
    def get_index_cls(cls) -> t.Type[CLIPFlatFloatIndex]:
        return CLIPFlatFloatIndex

    @classmethod
    def validate_signal_str(cls, signal_str: str) -> str:
        """
        Validate that the signal string represents a valid CLIP embedding.
        The signal string should be a hex-encoded float array.
        """
        try:
            # Try to decode as hex and convert to float array
            embedding = cls.deserialize_embedding(signal_str)
            if embedding.shape[0] != 512:
                raise ValueError(f"CLIP embeddings must be 512-dimensional. Got {embedding.shape[0]}")
            return signal_str
        except Exception as e:
            raise ValueError(f"Invalid CLIP signal string: {e}")

    @classmethod
    def hash_from_bytes(cls, bytes_: bytes) -> str:
        """
        Generate a CLIP embedding from a bytes object and serialize it.
        """
        clip_output = CLIP_HASHER.hash_from_bytes(bytes_)
        return cls.serialize_embedding(clip_output.hash_vector)

    @classmethod
    def compare_hash(
        cls, hash1: str, hash2: str, threshold: float = CLIP_DISTANCE_THRESHOLD
    ) -> signal_base.SignalComparisonResult:
        """
        Compare two CLIP embeddings using cosine distance.
        """
        vec1 = cls.deserialize_embedding(hash1)
        vec2 = cls.deserialize_embedding(hash2)
        distance: float = cosine_distance(vec1, vec2)
        return signal_base.SignalComparisonResult.from_simple_dist(distance, threshold)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> str:
        """
        Serialize a CLIP embedding to a hex string.
        """
        import binascii
        return binascii.hexlify(embedding.tobytes()).decode('ascii')

    @staticmethod
    def deserialize_embedding(signal_str: str) -> np.ndarray:
        """
        Deserialize a CLIP embedding from a hex string.
        """
        import binascii
        bytes_data = binascii.unhexlify(signal_str.encode('ascii'))
        return np.frombuffer(bytes_data, dtype=np.float32)

    @staticmethod
    def get_examples() -> t.List[str]:
        """
        Return example CLIP embeddings (simplified for brevity).
        In practice, these would be real CLIP embeddings from sample images.
        """
        # Create a sample normalized embedding
        sample_embedding = np.random.randn(512).astype(np.float32)
        sample_embedding = sample_embedding / np.linalg.norm(sample_embedding)
        return [CLIPSignal.serialize_embedding(sample_embedding)]


class TrivialCLIPIndex(signal_base.TrivialLinearSearchHashIndex):
    _SIGNAL_TYPE = CLIPSignal
