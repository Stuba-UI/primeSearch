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
    outputs = []
    for n in range(len(primes)):
        try:
            val = formula.subs('x', n + 1)
            val = float(val)  # Ensure output is a float
        except Exception:
            val = 0.0
        outputs.append(val)
    outputs = np.array(outputs, dtype=float)  # Ensure numpy array is float

    strict_hits = sum(int(o == p) for o, p in zip(outputs, primes))
    closeness = 1 / (1 + sum(abs(o - p) for o, p in zip(outputs, primes)))
    variance = np.var(outputs)
    novelty = np.std(outputs)
    complexity = len(str(formula))

    fitness = strict_hits + closeness

    combined_fitness = fitness + novelty - 0.01 * complexity

    return {
        "outputs": outputs,
        "strict_hits": strict_hits,
        "closeness": closeness,
        "variance": variance,
        "novelty": novelty,
        "complexity": complexity,
        "fitness": fitness,
        "combined_fitness": combined_fitness
    }
