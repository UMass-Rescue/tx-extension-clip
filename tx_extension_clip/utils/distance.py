import numpy as np
from scipy import spatial


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Returns the cosine distance of two vectors.
    Args:
        vector_a (np.ndarray): A vector of floats
        vector_b (np.ndarray): A vector of floats
    Returns:
        (float) The cosine distance of the two vectors.
    """
    return spatial.distance.cosine(vector_a, vector_b)


def hamming_distance(hash1: bytes, hash2: bytes) -> int:
    """
    Returns the hamming distance of two hashes.
    """
    h1 = np.unpackbits(np.frombuffer(hash1, dtype=np.uint8))
    h2 = np.unpackbits(np.frombuffer(hash2, dtype=np.uint8))
    # scipy's hamming distance returns a proportion, so we multiply by the
    # total number of bits to get the absolute distance.
    return int(spatial.distance.hamming(h1, h2) * h1.size)
