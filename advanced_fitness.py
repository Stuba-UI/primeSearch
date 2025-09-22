"""
advanced_fitness.py
-------------------
Evolution-friendly diagnostics for primeSearch.

Changes:
- Smooth non-integer penalty instead of harsh cutoff
- Normalize all diagnostic contributions
- Weighted combined fitness score
- Encourages exploration of more complex formulas
"""

import numpy as np

def diagnostics(formula, primes):
    """
    Returns a dict of diagnostic scores for a formula.
    """
    results = {
        "strict_hits": 0,
        "closeness": 0.0,
        "non_integer_penalty": 0.0,
        "negative_penalty": 0.0,
        "variance": 0.0,
        "novelty": 0.0,
        "combined_fitness": 0.0,
    }

    outputs = []

    for i, target_prime in enumerate(primes):
        try:
            val = formula.subs('x', i + 1)
            val_float = float(val)
            val_int = int(round(val_float))

            # Strict correctness
            if val_int == target_prime:
                results["strict_hits"] += 1

            # Closeness (reward smaller error)
            error = abs(val_float - target_prime)
            results["closeness"] += 1 / (1 + error)  # closer -> higher

            # Smooth non-integer penalty
            frac_part = abs(val_float - round(val_float))
            results["non_integer_penalty"] -= frac_part * 0.5  # gentle penalty

            # Negative penalty
            if val_float < 0:
                results["negative_penalty"] -= 0.2

            outputs.append(val_float)

        except Exception:
            # Penalize formulas that fail
            results["non_integer_penalty"] -= 0.5
            continue

    if outputs:
        results["variance"] = np.var(outputs)
        results["novelty"] = len(set(outputs)) / len(outputs)

        # Normalize diagnostics roughly to same scale
        strict_hits_norm = results["strict_hits"] / len(primes)
        closeness_norm = results["closeness"] / len(primes)
        novelty_norm = results["novelty"]
        variance_norm = results["variance"] / (max(outputs) ** 2 + 1e-6)
    else:
        # if outputs is empty, everything is zero or negative penalty
        strict_hits_norm = 0.0
        closeness_norm = 0.0
        novelty_norm = 0.0
        variance_norm = 0.0

    # Weighted combined fitness
    results["combined_fitness"] = (
        2.0 * strict_hits_norm +
        1.0 * closeness_norm +
        2.0 * novelty_norm +  # Increase novelty weight
        0.2 * variance_norm +
        results["non_integer_penalty"] +
        results["negative_penalty"]
    )

    return results
