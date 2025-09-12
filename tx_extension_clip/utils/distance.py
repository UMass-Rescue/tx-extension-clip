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
    h1_array = np.frombuffer(hash1, dtype=np.uint8)
    h2_array = np.frombuffer(hash2, dtype=np.uint8)
    
    xor_result = h1_array ^ h2_array
    
    return int(np.sum(np.unpackbits(xor_result)))
