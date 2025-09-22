import sympy
from sympy import symbols
import random

x = symbols('x')

# ----------------------------
# Random formula (linear or quadratic)
# ----------------------------
def random_formula(max_add=50, max_mul=10, max_quad=5):
    """Generate a random linear or quadratic formula."""
    if random.random() < 0.3:
        k = random.randint(1, max_quad)
        m = random.randint(1, max_mul)
        n = random.randint(1, max_add)
        return k*x**2 + m*x + n
    else:
        k = random.randint(1, max_mul)
        n = random.randint(1, max_add)
        return k*x + n

# ----------------------------
# Safe evaluation
# ----------------------------
def evaluate_formula(formula, n):
    try:
        val = formula.subs(x, n)
        if val.is_number:
            return int(val)
        return None
    except Exception:
        return None

# ----------------------------
# Mutation (coefficient adjustment)
# ----------------------------
def mutate_formula(formula, max_mutation=5):
    if formula is None:
        return random_formula()

    try:
        expr = sympy.expand(formula)
        # Extract coefficients
        coeffs = {term: int(expr.coeff(term)) for term in [x**2, x, 1]}
        # Apply mutation
        for term in coeffs:
            if coeffs[term] != 0:
                coeffs[term] += random.randint(-max_mutation, max_mutation)
                if term != 1:
                    coeffs[term] = max(1, coeffs[term])  # keep multiplier positive
        # Reconstruct formula
        new_formula = coeffs[x**2]*x**2 + coeffs[x]*x + coeffs[1]
        return new_formula
    except Exception:
        # fallback to previous formula (do NOT reset completely)
        return formula

# ----------------------------
# Crossover (blend parent coefficients)
# ----------------------------
def crossover_formula(f1, f2):
    if f1 is None:
        f1 = random_formula()
    if f2 is None:
        f2 = random_formula()
    try:
        expr1 = sympy.expand(f1)
        expr2 = sympy.expand(f2)
        # Extract coefficients
        coeffs1 = {term: int(expr1.coeff(term)) for term in [x**2, x, 1]}
        coeffs2 = {term: int(expr2.coeff(term)) for term in [x**2, x, 1]}
        # Blend coefficients randomly
        new_coeffs = {}
        for term in coeffs1:
            new_coeffs[term] = coeffs1[term] if random.random() < 0.5 else coeffs2[term]
            # Add small noise
            new_coeffs[term] += random.randint(-2, 2)
            if term != 1:
                new_coeffs[term] = max(1, new_coeffs[term])
        # Reconstruct formula
        return new_coeffs[x**2]*x**2 + new_coeffs[x]*x + new_coeffs[1]
    except Exception:
        return f1  # fallback to first parent
