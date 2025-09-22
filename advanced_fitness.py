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
            val = float(val)
        except Exception:
            val = 0.0
        outputs.append(val)
    outputs = np.array(outputs, dtype=float)

    # Penalize outputs that are negative or not close to integer
    non_integer_penalty = np.sum(np.abs(outputs - np.round(outputs)))
    negative_penalty = np.sum(outputs < 0)

    # Main metrics
    strict_hits = sum(int(np.isclose(o, p, atol=1e-6)) for o, p in zip(outputs, primes))
    near_hits = sum(int(np.abs(o - p) < 2) for o, p in zip(outputs, primes))
    mse = np.mean((outputs - primes) ** 2)
    # Log-transform the MSE for a smoother, more meaningful fitness landscape
    closeness = -np.log1p(mse)

    variance = np.var(outputs)
    novelty = np.mean(np.abs(np.diff(outputs)))
    novelty = min(novelty, 1e4)
    complexity = len(str(formula))

    # Fitness: prioritize closeness, allow complexity, reward hits
    fitness = (
        1000 * closeness +      # Log-MSE: higher is better, more sensitive
        1000 * strict_hits +
        100 * near_hits -
        0.1 * non_integer_penalty -
        10 * negative_penalty +
        0.01 * novelty
    )
    combined_fitness = fitness - 0.0001 * complexity  # Even lower complexity penalty

    return {
        "outputs": outputs,
        "strict_hits": strict_hits,
        "near_hits": near_hits,
        "closeness": closeness,
        "variance": variance,
        "novelty": novelty,
        "complexity": complexity,
        "fitness": fitness,
        "combined_fitness": combined_fitness
    }
