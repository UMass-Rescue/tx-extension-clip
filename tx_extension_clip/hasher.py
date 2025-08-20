"""
CLIP hashing utilities. Includes a CLIPHasher class that handles generating the CLIP hashes.
"""
from __future__ import annotations

import binascii
import io
import pathlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from open_clip.factory import (
    convert_to_custom_text_state_dict,
    load_state_dict,
    resize_pos_embed,
)
from PIL import Image


def binarize_clip_embeddings(embeddings, nbits=512, seed=42):
    """
    Binarize CLIP embeddings using random hyperplane LSH.
    
    Args:
        embeddings: numpy array of shape (N, d) with CLIP vectors (float32).
        nbits: how many bits per code (default 512 to match CLIP dimension).
        seed: random seed for reproducible hyperplanes.
        
    Returns:
        Binary codes of shape (N, nbits) with values {0, 1}
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Generate random hyperplanes
    R = np.random.randn(d, nbits).astype(np.float32)

    # Project embeddings onto hyperplanes
    proj = embeddings @ R

    # Binarize to {0,1} based on sign
    codes = (proj >= 0).astype(np.uint8)
    return codes


def binary_quantize_embedding(embedding: np.ndarray, nbits: int = 512) -> np.ndarray:
    """
    Convert CLIP float embeddings to binary representation using LSH.
    
    Args:
        embedding: Float32 CLIP embedding vector (1D)
        nbits: Number of bits for output
        
    Returns:
        Binary array (uint8) where each element is 0 or 1
    """
    # Reshape to (1, d) for compatibility with binarize_clip_embeddings
    embedding_2d = embedding.reshape(1, -1)
    binary_codes = binarize_clip_embeddings(embedding_2d, nbits=nbits, seed=42)
    return binary_codes[0]  # Return first (and only) row


def pack_bits(codes):
    """
    Pack binary codes for FAISS binary index.
    codes: array of shape (N, nbits), dtype=uint8 {0,1}
    returns: packed array of shape (N, nbits//8), dtype=uint8
    """
    return np.packbits(codes, axis=1)


def pack_binary_to_bytes(binary_array: np.ndarray) -> bytes:
    """
    Pack binary array into bytes for efficient storage.
    
    Args:
        binary_array: Array of 0s and 1s (1D)
        
    Returns:
        Packed bytes representation
    """
    # Pad to multiple of 8 if needed
    remainder = len(binary_array) % 8
    if remainder != 0:
        padding = 8 - remainder
        binary_array = np.concatenate([binary_array, np.zeros(padding, dtype=np.uint8)])
    
    # Pack 8 bits into each byte
    packed = np.packbits(binary_array)
    return packed.tobytes()


@dataclass
class CLIPOutput:
    """
    The output of the CLIP hasher.

    We need to make sure to keep the `model_name` and `pretrained` to specify which version
    of the model was used.
    """

    model_name: str
    pretrained: str
    normalized: bool
    hash_vector: Optional[np.ndarray] = None
    binary_hash: Optional[np.ndarray] = None

    def deserialize(self, hex_: str) -> CLIPOutput:
        """Deserializes the CLIP hash from a string.

        Args:
            hex_ (str) : The serialized CLIP hash.

        Returns:
            CLIPOutput: The deserialized CLIP hash.
        """
        bytes_: bytes = binascii.unhexlify(bytes(hex_, "ascii"))
        hash_vector: np.ndarray = np.frombuffer(bytes_, dtype=np.float32)
        return CLIPOutput(
            hash_vector=hash_vector,
            model_name=self.model_name,
            pretrained=self.pretrained,
            normalized=self.normalized,
        )

    def serialize(self) -> str:
        """Serializes the CLIP hash to a string.

        Returns:
            str: The serialized CLIP hash.
        """
        if self.hash_vector is not None:
            # Fallback to float vector serialization
            return str(binascii.hexlify(self.hash_vector.tobytes()), "ascii")
        elif self.binary_hash is not None:
            # Serialize binary hash (preferred for indexing)
            # Pack bits for FAISS binary index compatibility
            binary_2d = self.binary_hash.reshape(1, -1)  # Shape: (1, nbits)
            packed_binary = pack_bits(binary_2d)[0]  # Shape: (nbits//8,)
            return str(binascii.hexlify(packed_binary.tobytes()), "ascii")
        else:
            raise ValueError("Both hash_vector and binary_hash are None")


class CLIPHasher:
    """
    The CLIP hasher. Handles the CLIP model and transform pipeline to generate hashes.
    """

    def __init__(self, model_name: str, pretrained: str, normalized: bool = True):
        self.model_name: str = model_name
        self.pretrained: str = pretrained
        self.normalized: bool = normalized

        self._model: Optional[torch.nn.Module] = None
        self._transform: Optional[transforms.Compose] = None

        #this open clip torch issue has been fixed with this merge  https://github.com/mlfoundations/open_clip/pull/595
        #fix_open_clip()

    @property
    def model(self) -> torch.nn.Module:
        """Returns the CLIP model."""
        if self._model is None:
            self.init_model_and_transforms()
        return self._model

    @property
    def transform(self) -> transforms.Compose:
        """Returns the CLIP image transform pipeline."""
        if self._transform is None:
            self.init_model_and_transforms()
        return self._transform

    def init_model_and_transforms(self):
        """Initializes the CLIP model and transform pipeline."""

        # we do not need the training transformations
        self._model, _, self._transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )

    def hash_from_file(self, path: pathlib.Path) -> CLIPOutput:
        """Returns the CLIP hash from a file path.

        Args:
            path (pathlib.Path): The path to the file.

        Returns:
            CLIPOutput: The CLIP hash.
        """
        return self.hash_from_file_list([path])[0]

    def hash_from_bytes(self, file_bytes: bytes) -> CLIPOutput:
        """Returns the CLIP hash from a bytes object.

        Args:
            file_bytes (bytes): The bytes object.

        Returns:
            CLIPOutput: The CLIP hash.
        """
        return self.hash_from_bytes_list([file_bytes])[0]

    def hash_from_image(self, image: Image) -> CLIPOutput:
        """Returns the CLIP hash from a PIL Image object.

        Args:
            image (Image): The PIL Image object.

        Returns:
            CLIPOutput: The CLIP hash.
        """
        return self.hash_from_image_list([image])

    def hash_from_file_list(self, paths: List[pathlib.Path]) -> List[CLIPOutput]:
        """Returns the CLIP hash from a list of file paths.

        Args:
            paths (List[pathlib.Path]): The list of file paths.

        Returns:
            List[CLIPOutput]: The CLIP hashes.
        """
        images: List[Image.Image] = [Image.open(path) for path in paths]
        return self.hash_from_image_list(images)

    def hash_from_image_list(self, images: List[Image.Image]) -> List[CLIPOutput]:
        """Returns the CLIP hash from a list of PIL Image objects.

        Args:
            images (List[Image]): The list of PIL Image objects.

        Returns:
            List[CLIPOutput]: The CLIP hashes.
        """
        transformed_images: torch.Tensor = torch.stack(
            [self.transform(image) for image in images]
        )
        with torch.no_grad():
            image_features: torch.Tensor = self.model.visual(transformed_images)
        if self.normalized:
            image_features = torch.nn.functional.normalize(image_features, dim=1)
        return [
            self._create_clip_output(image_feature.numpy())
            for image_feature in image_features
        ]
    
    def _create_clip_output(self, float_vector: np.ndarray) -> CLIPOutput:
        """Create CLIPOutput with both float and binary representations."""
        binary_hash = binary_quantize_embedding(float_vector, nbits=512)
        
        return CLIPOutput(
            hash_vector=float_vector,
            binary_hash=binary_hash,
            model_name=self.model_name,
            pretrained=self.pretrained,
            normalized=self.normalized,
        )

    def hash_from_bytes_list(self, file_bytes_list: List[bytes]) -> List[CLIPOutput]:
        """Returns the CLIP hash from a list of bytes objects.

        Args:
            file_bytes_list (List[bytes]): The list of bytes objects.

        Returns:
            List[CLIPOutput]: The CLIP hashes.
        """
        images: List[Image.Image] = [
            Image.open(io.BytesIO(file_bytes)) for file_bytes in file_bytes_list
        ]
        return self.hash_from_image_list(images)

    def get_version_str(self) -> str:
        """Returns a string representing the version of the model used."""
        return f"{self.model_name}-{self.pretrained}-{'normalized' if self.normalized else 'unnormalized'}"


def fix_open_clip():
    """
    This is a fix for an incompatibility between a new version of `transformers` and `open_clip`.
    I hate it.
    Info here: https://github.com/mlfoundations/open_clip/pull/595
    The above PR is merged, but not yet released as of writing this.
    """

    def _load_checkpoint(model, checkpoint_path, strict=True):
        state_dict = load_state_dict(checkpoint_path)
        if "positional_embedding" in state_dict and not hasattr(
            model, "positional_embedding"
        ):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        resize_pos_embed(state_dict, model)
        del state_dict["text.transformer.embeddings.position_ids"]
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        return incompatible_keys

    open_clip.factory.load_checkpoint = _load_checkpoint
