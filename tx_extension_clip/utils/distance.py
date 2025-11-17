import numpy as np
from scipy import spatial


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Returns the cosine distance of two vectors.
    
    Handles edge cases where scipy.spatial.distance.cosine returns NaN:
    - Both vectors are zero → distance = 0.0 (identical)
    - One vector is zero → distance = 1.0 (maximum dissimilarity)
    - Other NaN cases → distance = 1.0 (treat as no similarity)
    
    Args:
        vector_a (np.ndarray): A vector of floats
        vector_b (np.ndarray): A vector of floats
    Returns:
        (float) The cosine distance of the two vectors (0.0 to 1.0).
                Always returns a Python native float for JSON serialization.
    """
    # Check for zero vectors first to avoid NaN from scipy
    a_is_zero = np.allclose(vector_a, 0, rtol=1e-9, atol=1e-9)
    b_is_zero = np.allclose(vector_b, 0, rtol=1e-9, atol=1e-9)
    
    if a_is_zero and b_is_zero:
        # Both are zero vectors → identical → distance = 0
        return 0.0
    elif a_is_zero or b_is_zero:
        # Only one is zero → completely dissimilar → distance = 1
        return 1.0
    
    # Compute cosine distance for non-zero vectors
    distance = spatial.distance.cosine(vector_a, vector_b)
    
    # Handle any remaining NaN cases (shouldn't happen after above checks)
    if np.isnan(distance):
        distance = 1.0  # Treat unexpected NaN as maximum distance
    
    # Convert numpy type to Python native float for JSON serialization
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
