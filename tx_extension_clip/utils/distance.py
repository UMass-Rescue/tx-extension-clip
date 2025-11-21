import numpy as np
from scipy import spatial


def similarity_to_distance(similarity: float) -> float:
    """Converts a cosinesimilarity score to a distance metric.

    Parameters:
        similarity (float): Similarity value, typically in the range [-1.0, 1.0].
    
    Returns:
        float: Distance value.
    """
    return float(1.0 - similarity)


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Returns cosine distance between two vectors, handling edge cases for zero vectors.
    
    For normalized vectors, cosine distance ranges from [0, 2]:
    - 0.0 = identical vectors (cosine similarity = 1.0)
    - 1.0 = orthogonal vectors (cosine similarity = 0.0)
    - 2.0 = opposite vectors (cosine similarity = -1.0)
    
    Uses scipy's cosine distance which returns 1 - cosine_similarity.
    For normalized vectors, this gives the full range [0, 2].
    """
    a_is_zero = np.allclose(vector_a, 0, rtol=1e-9, atol=1e-9)
    b_is_zero = np.allclose(vector_b, 0, rtol=1e-9, atol=1e-9)
    
    if a_is_zero and b_is_zero:
        return 0.0
    elif a_is_zero or b_is_zero:
        return 1.0
    
    distance = spatial.distance.cosine(vector_a, vector_b)
    
    if np.isnan(distance):
        distance = 1.0
    
    return float(distance)


def hamming_distance(hash1: bytes, hash2: bytes) -> int:
    """
    Returns the hamming distance of two hashes.
    Optimized version using bitwise XOR for better performance.
    """
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must be of the same length")
    h1_array = np.frombuffer(hash1, dtype=np.uint8)
    h2_array = np.frombuffer(hash2, dtype=np.uint8)
    
    xor_result = h1_array ^ h2_array
    
    return int(np.sum(np.unpackbits(xor_result)))
