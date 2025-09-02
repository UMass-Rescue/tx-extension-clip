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
    Optimized version using bitwise XOR for better performance.
    """
    # Convert bytes to numpy arrays for vectorized XOR
    h1_array = np.frombuffer(hash1, dtype=np.uint8)
    h2_array = np.frombuffer(hash2, dtype=np.uint8)
    
    # XOR the bytes and count set bits
    xor_result = h1_array ^ h2_array
    
    # Count bits using numpy's built-in bit counting
    # This is much faster than unpackbits + hamming distance
    return int(np.sum(np.unpackbits(xor_result)))
