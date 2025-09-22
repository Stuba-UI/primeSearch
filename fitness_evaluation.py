"""
fitness_evaluation.py
--------------------
Integer-safe prime-focused fitness for primeSearch.

Features:
- Evaluates symbolic formulas against actual prime sequences
- Returns higher fitness for formulas that correctly predict primes
- Robust to formula errors and non-integer outputs
"""

from sympy import isprime
import csv
from .prime_data import get_primes
from .symbolic_formula import generate_random_formula
from .advanced_fitness import diagnostics

def fitness(formula, primes):
    """
    Compute strict fitness of a formula:
    Fitness = number of primes correctly generated in sequence
    """
    score = 0
    for i, target_prime in enumerate(primes):
        try:
            val = formula.subs('x', i + 1)  # index = n
            val_int = int(val)
            if val_int == target_prime:
                score += 1
        except Exception:
            continue
    return score

def evaluate_population(population, primes, csv_path="evolution_log.csv"):
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["formula", "strict_hits", "near_hits", "closeness", "variance", "novelty", "complexity", "fitness", "combined_fitness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for formula in population:
            diag = diagnostics(formula, primes)
            row = {k: diag[k] for k in fieldnames if k in diag}
            row["formula"] = str(formula)
            writer.writerow(row)

# --- Prime generator ---
def generate_primes(n):
    """Generate first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        for p in primes:
            if candidate % p == 0:
                break
        else:
            primes.append(candidate)
        candidate += 1
    return primes

# Default prime list (first 100 primes)
primes = generate_primes(100)
