"""
prime_data.py
-------------
Module responsible for prime dataset generation, feature extraction,
and embedding construction for neural-symbolic co-evolution.

Project: primeSearch
Author: Tuomas Lehto
Purpose: Provide structured numeric representations to guide symbolic 
formula evolution for universal prime pattern discovery.

Note: Designed for progressive scaling to very large primes.
"""

import sympy
import numpy as np
import torch

# ----------------------------
# 1. Prime dataset generation
# ----------------------------
def generate_primes(limit=10000):
    """
    Generate all prime numbers up to 'limit'.

    Args:
        limit (int): Upper bound for prime generation.

    Returns:
        List[int]: Ordered list of prime numbers.
    """
    return list(sympy.primerange(2, limit))

# ----------------------------
# 2. Feature extraction
# ----------------------------
def extract_features(n):
    """
    Convert integer n into a structured feature vector for neural guidance.

    Features:
        - Modular residues (mod 2,3,5,7)
        - Digit sum
        - Last digit
        - Placeholder for prime gap

    Args:
        n (int): Prime number to encode.

    Returns:
        np.ndarray: Feature vector as float array.
    """
    digits = [int(d) for d in str(n)]
    return np.array([
        n % 2,
        n % 3,
        n % 5,
        n % 7,
        sum(digits),
        digits[-1] if digits else 0
    ], dtype=float)

def generate_features(primes):
    """
    Generate a feature matrix for a list of primes.

    Args:
        primes (List[int]): Ordered prime numbers.

    Returns:
        np.ndarray: Feature matrix (num_primes x num_features)
    """
    return np.array([extract_features(p) for p in primes])

# ----------------------------
# 3. Torch tensor conversion
# ----------------------------
def features_to_tensor(features):
    """
    Convert numpy feature matrix into PyTorch tensor.

    Args:
        features (np.ndarray): Feature matrix.

    Returns:
        torch.Tensor: Float32 tensor for NN input.
    """
    return torch.tensor(features, dtype=torch.float32)

# ----------------------------
# 4. Example usage
# ----------------------------
if __name__ == "__main__":
    primes = generate_primes(10000)
    features = generate_features(primes)
    tensor_features = features_to_tensor(features)
    print("Example prime features (first 5):")
    print(features[:5])
